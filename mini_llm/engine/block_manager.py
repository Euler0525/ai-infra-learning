from collections import deque
from dataclasses import dataclass

from mini_llm.engine.sequence import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16):
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [Block(block_id) for block_id in range(num_blocks)]
        self.free_blocks = deque(self.blocks)
        self.used_blocks: set[int] = set()
        self._allocated_request_ids: set[str] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return len(self.used_blocks)

    def can_allocate(self, sequence: Sequence) -> bool:
        self._check_block_size(sequence)
        return (
            sequence.request_id not in self._allocated_request_ids
            and not sequence.block_table
            and sequence.num_blocks <= self.num_free_blocks
        )

    def allocate(self, sequence: Sequence) -> None:
        self._check_block_size(sequence)
        if sequence.request_id in self._allocated_request_ids:
            raise ValueError("sequence is already allocated")
        if sequence.block_table:
            raise ValueError("sequence block table must be empty before allocation")
        if sequence.num_blocks > self.num_free_blocks:
            raise RuntimeError("not enough free blocks")

        for _ in range(sequence.num_blocks):
            sequence.block_table.append(self._allocate_block())
        self._allocated_request_ids.add(sequence.request_id)

    def can_append(self, sequence: Sequence) -> bool:
        self._check_block_size(sequence)
        if sequence.request_id not in self._allocated_request_ids:
            return False
        additional_blocks = max(0, sequence.num_blocks - len(sequence.block_table))
        return additional_blocks <= self.num_free_blocks

    def append(self, sequence: Sequence) -> None:
        self._check_block_size(sequence)
        if sequence.request_id not in self._allocated_request_ids:
            raise ValueError("sequence is not allocated")
        if not self.can_append(sequence):
            raise RuntimeError("not enough free blocks to append a token")

        while len(sequence.block_table) < sequence.num_blocks:
            sequence.block_table.append(self._allocate_block())

    def deallocate(self, sequence: Sequence) -> None:
        if sequence.request_id not in self._allocated_request_ids:
            raise ValueError("sequence is not allocated or was already released")

        for block_id in sequence.block_table:
            self._release_block(block_id)
        sequence.block_table.clear()
        sequence.num_cached_tokens = 0
        self._allocated_request_ids.remove(sequence.request_id)

    def _allocate_block(self) -> int:
        block = self.free_blocks.popleft()
        if block.ref_count != 0:
            raise RuntimeError("free block has a non-zero reference count")
        block.ref_count = 1
        self.used_blocks.add(block.block_id)
        return block.block_id

    def _release_block(self, block_id: int) -> None:
        if block_id not in self.used_blocks:
            raise ValueError(f"block {block_id} is not allocated")
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.used_blocks.remove(block_id)
            self.free_blocks.append(block)

    def _check_block_size(self, sequence: Sequence) -> None:
        if sequence.block_size != self.block_size:
            raise ValueError("sequence and block manager block sizes must match")
