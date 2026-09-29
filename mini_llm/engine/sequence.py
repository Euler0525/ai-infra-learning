from dataclasses import dataclass, field
from enum import Enum

from mini_llm.config.sampling import SamplingParams


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_size: int = 16
    generated_token_ids: list[int] = field(default_factory=list, init=False)
    block_table: list[int] = field(default_factory=list)
    num_cached_tokens: int = 0
    _status: SequenceStatus = field(
        default=SequenceStatus.WAITING,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.prompt_token_ids = list(self.prompt_token_ids)
        self.block_table = list(self.block_table)

    @property
    def status(self) -> SequenceStatus:
        return self._status

    def transition_to(self, status: SequenceStatus) -> None:
        next_status = {
            SequenceStatus.WAITING: SequenceStatus.RUNNING,
            SequenceStatus.RUNNING: SequenceStatus.FINISHED,
        }.get(self._status)
        if status is not next_status:
            raise ValueError(
                f"invalid sequence status transition: "
                f"{self._status.value} -> {status.value}"
            )
        self._status = status

    def append_token(self, token_id: int) -> None:
        if self._status is not SequenceStatus.RUNNING:
            raise RuntimeError("tokens can only be appended to a running sequence")
        self.generated_token_ids.append(token_id)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_completion_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_completion_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def num_blocks(self) -> int:
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        if self.num_tokens == 0:
            return 0
        return (self.num_tokens - 1) % self.block_size + 1

    @property
    def is_finished(self) -> bool:
        return self._status is SequenceStatus.FINISHED
