import pytest

from mini_llm.config import SamplingParams
from mini_llm.engine import Sequence, SequenceStatus


@pytest.mark.parametrize(
    ("num_tokens", "num_blocks", "last_block_num_tokens"),
    [
        (0, 0, 0),
        (1, 1, 1),
        (15, 1, 15),
        (16, 1, 16),
        (17, 2, 1),
        (31, 2, 15),
        (32, 2, 16),
        (33, 3, 1),
    ],
)
def test_block_boundaries(
    num_tokens: int,
    num_blocks: int,
    last_block_num_tokens: int,
) -> None:
    sequence = Sequence(
        request_id="request-1",
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(),
    )

    assert sequence.num_tokens == num_tokens
    assert sequence.num_blocks == num_blocks
    assert sequence.last_block_num_tokens == last_block_num_tokens


def test_copies_input_lists_and_owns_default_block_table() -> None:
    prompt_token_ids = [1, 2, 3]
    block_table = [4]
    first = Sequence(
        request_id="request-1",
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(),
        block_table=block_table,
    )
    second = Sequence(
        request_id="request-2",
        prompt_token_ids=[],
        sampling_params=SamplingParams(),
    )

    prompt_token_ids.append(4)
    block_table.append(5)
    first.block_table.append(6)

    assert first.prompt_token_ids == [1, 2, 3]
    assert first.block_table == [4, 6]
    assert second.block_table == []


def test_prompt_and_completion_tokens_remain_separate() -> None:
    params = SamplingParams(max_tokens=2)
    sequence = Sequence("request-1", [10, 11, 12], params)
    sequence.transition_to(SequenceStatus.RUNNING)
    sequence.append_token(20)
    sequence.append_token(21)

    assert sequence.request_id == "request-1"
    assert sequence.sampling_params is params
    assert sequence.prompt_token_ids == [10, 11, 12]
    assert sequence.generated_token_ids == [20, 21]
    assert sequence.num_prompt_tokens == 3
    assert sequence.num_completion_tokens == 2
    assert sequence.num_tokens == 5


def test_tracks_cache_and_block_state() -> None:
    sequence = Sequence("request-1", [1, 2, 3], SamplingParams())

    sequence.block_table.extend([7, 2])
    sequence.num_cached_tokens = 3

    assert sequence.block_table == [7, 2]
    assert sequence.num_cached_tokens == 3


def test_allows_only_forward_status_transitions() -> None:
    sequence = Sequence("request-1", [], SamplingParams())

    assert sequence.status is SequenceStatus.WAITING
    assert not sequence.is_finished

    sequence.transition_to(SequenceStatus.RUNNING)
    assert sequence.status is SequenceStatus.RUNNING

    sequence.transition_to(SequenceStatus.FINISHED)
    assert sequence.status is SequenceStatus.FINISHED
    assert sequence.is_finished

    with pytest.raises(ValueError, match="finished -> running"):
        sequence.transition_to(SequenceStatus.RUNNING)


@pytest.mark.parametrize(
    ("initial_status", "target_status"),
    [
        (SequenceStatus.WAITING, SequenceStatus.FINISHED),
        (SequenceStatus.WAITING, SequenceStatus.WAITING),
        (SequenceStatus.RUNNING, SequenceStatus.WAITING),
        (SequenceStatus.RUNNING, SequenceStatus.RUNNING),
    ],
)
def test_rejects_invalid_status_transitions(
    initial_status: SequenceStatus,
    target_status: SequenceStatus,
) -> None:
    sequence = Sequence("request-1", [], SamplingParams())
    if initial_status is SequenceStatus.RUNNING:
        sequence.transition_to(SequenceStatus.RUNNING)

    with pytest.raises(ValueError, match="invalid sequence status transition"):
        sequence.transition_to(target_status)


@pytest.mark.parametrize("status", [SequenceStatus.WAITING, SequenceStatus.FINISHED])
def test_append_token_requires_running_status(status: SequenceStatus) -> None:
    sequence = Sequence("request-1", [], SamplingParams())
    if status is SequenceStatus.FINISHED:
        sequence.transition_to(SequenceStatus.RUNNING)
        sequence.transition_to(SequenceStatus.FINISHED)

    with pytest.raises(RuntimeError, match="running sequence"):
        sequence.append_token(1)
