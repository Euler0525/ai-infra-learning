import pytest

from mini_llm.config import SamplingParams
from mini_llm.engine import BlockManager, Sequence, SequenceStatus


def make_sequence(request_id: str, num_tokens: int) -> Sequence:
    return Sequence(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(),
    )


def assert_manager_invariants(manager: BlockManager) -> None:
    free_ids = {block.block_id for block in manager.free_blocks}
    assert manager.num_free_blocks + manager.num_used_blocks == manager.num_blocks
    assert free_ids.isdisjoint(manager.used_blocks)
    assert free_ids | manager.used_blocks == set(range(manager.num_blocks))
    for block in manager.blocks:
        expected_ref_count = 1 if block.block_id in manager.used_blocks else 0
        assert block.ref_count == expected_ref_count


@pytest.mark.parametrize(
    ("num_tokens", "expected_blocks"),
    [
        (0, 0),
        (1, 1),
        (15, 1),
        (16, 1),
        (17, 2),
        (31, 2),
        (32, 2),
        (33, 3),
    ],
)
def test_allocate_respects_block_boundaries(
    num_tokens: int,
    expected_blocks: int,
) -> None:
    manager = BlockManager(num_blocks=4)
    sequence = make_sequence("request-1", num_tokens)

    assert manager.can_allocate(sequence)
    manager.allocate(sequence)

    assert len(sequence.block_table) == expected_blocks
    assert manager.num_used_blocks == expected_blocks
    assert_manager_invariants(manager)

    manager.deallocate(sequence)
    assert sequence.block_table == []
    assert all(block.ref_count == 0 for block in manager.blocks)
    assert_manager_invariants(manager)


@pytest.mark.parametrize(
    ("num_tokens", "expected_blocks_after_append"),
    [
        (0, 0),
        (1, 1),
        (15, 1),
        (16, 1),
        (17, 2),
        (31, 2),
        (32, 2),
        (33, 3),
    ],
)
def test_append_allocates_only_when_crossing_a_boundary(
    num_tokens: int,
    expected_blocks_after_append: int,
) -> None:
    manager = BlockManager(num_blocks=4)
    sequence = make_sequence("request-1", num_tokens)
    manager.allocate(sequence)

    manager.append(sequence)

    assert len(sequence.block_table) == expected_blocks_after_append
    assert manager.num_used_blocks == expected_blocks_after_append
    assert_manager_invariants(manager)


def test_allocate_append_and_deallocate() -> None:
    manager = BlockManager(num_blocks=2)
    sequence = make_sequence("request-1", 16)
    manager.allocate(sequence)
    sequence.num_cached_tokens = 16
    sequence.transition_to(SequenceStatus.RUNNING)

    sequence.append_token(100)
    manager.append(sequence)

    assert sequence.num_tokens == 17
    assert sequence.block_table == [0, 1]
    assert manager.num_free_blocks == 0
    assert_manager_invariants(manager)

    manager.deallocate(sequence)

    assert sequence.block_table == []
    assert sequence.num_cached_tokens == 0
    assert manager.num_free_blocks == 2
    assert manager.used_blocks == set()
    assert all(block.ref_count == 0 for block in manager.blocks)
    assert_manager_invariants(manager)


def test_multiple_sequences_allocate_interleaved_blocks() -> None:
    manager = BlockManager(num_blocks=4)
    first = make_sequence("request-1", 17)
    second = make_sequence("request-2", 16)
    third = make_sequence("request-3", 1)

    manager.allocate(first)
    manager.allocate(second)
    second.transition_to(SequenceStatus.RUNNING)
    second.append_token(100)
    manager.append(second)

    assert first.block_table == [0, 1]
    assert second.block_table == [2, 3]
    assert not manager.can_allocate(third)

    manager.deallocate(first)
    assert manager.can_allocate(third)
    manager.allocate(third)

    assert third.block_table == [0]
    assert_manager_invariants(manager)

    manager.deallocate(second)
    manager.deallocate(third)
    assert all(block.ref_count == 0 for block in manager.blocks)
    assert_manager_invariants(manager)


def test_cache_exhaustion_is_reported_before_mutation() -> None:
    manager = BlockManager(num_blocks=2)
    full = make_sequence("request-1", 32)
    waiting = make_sequence("request-2", 1)
    manager.allocate(full)

    assert not manager.can_allocate(waiting)
    with pytest.raises(RuntimeError, match="not enough free blocks"):
        manager.allocate(waiting)
    assert waiting.block_table == []

    full.transition_to(SequenceStatus.RUNNING)
    full.append_token(100)
    assert not manager.can_append(full)
    with pytest.raises(RuntimeError, match="append a token"):
        manager.append(full)
    assert full.block_table == [0, 1]
    assert_manager_invariants(manager)


def test_repeated_deallocation_is_rejected() -> None:
    manager = BlockManager(num_blocks=1)
    sequence = make_sequence("request-1", 1)
    manager.allocate(sequence)
    manager.deallocate(sequence)

    with pytest.raises(ValueError, match="already released"):
        manager.deallocate(sequence)

    assert manager.num_free_blocks == 1
    assert manager.used_blocks == set()
    assert manager.blocks[0].ref_count == 0
    assert_manager_invariants(manager)
