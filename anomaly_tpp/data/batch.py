from typing import List, Optional

import torch

from .sequence import Sequence
from .utils import DotDict


class Batch(DotDict):
    """Batch of padded variable-length sequences.

    Args:
        arrival_times: arrival times, shape (batch_size, seq_len)
        inter_times: inter-event times, shape (batch_size, seq_len)
        t_max: length of the interval for each sequence, shape (batch_size)
        mask: boolean indicator for events (= not padding), shape (batch_size, seq_len)
        marks: shape categorical marks, shape (batch_size, seq_len)
    """

    def __init__(
        self,
        arrival_times: torch.Tensor,
        inter_times: torch.Tensor,
        t_max: torch.Tensor,
        mask: torch.Tensor,
        marks: Optional[torch.Tensor] = None,
    ):
        self.arrival_times = arrival_times
        self.inter_times = inter_times
        self.t_max = t_max
        self.mask = mask
        self.marks = marks

        self._validate()

    @property
    def batch_size(self):
        """Number of sequences in the batch."""
        return self.arrival_times.shape[0]

    @property
    def seq_len(self):
        """Length of the padded sequences."""
        return self.arrival_times.shape[1]

    def _validate(self):
        """Check if all tensors have correct shapes."""
        if self.arrival_times.ndim != 2:
            raise ValueError(
                f"arrival_times must be a 2-d tensor (got {self.arrival_times.ndim}-d)"
            )
        if self.t_max.shape != (self.batch_size,):
            raise ValueError(
                f"t_max must be of shape (batch_size,), got {self.t_max.shape}"
            )
        if self.mask.shape != (self.batch_size, self.seq_len):
            raise ValueError(
                f"mask must be of shape (batch_size={self.batch_size}, "
                f" max_seq_len={self.seq_len}), got {self.mask.shape}"
            )
        if self.marks is not None and self.marks.shape != (
            self.batch_size,
            self.seq_len,
        ):
            raise ValueError(
                f"marks must be of shape (batch_size={self.batch_size},"
                f" seq_len={self.seq_len}), got {self.marks.shape}"
            )

    @staticmethod
    def from_list(sequences: List[Sequence]) -> "Batch":
        """Construct a batch from a list of sequences."""
        batch_size = len(sequences)
        device = sequences[0].arrival_times.device

        t_max = torch.stack([s.t_max for s in sequences])  # (B)
        inter_times = pad_sequence([s.inter_times for s in sequences])  # (B, L)
        arrival_times = inter_times.cumsum(dim=-1)  # (B, L)

        padded_seq_len = inter_times.shape[1]
        mask = torch.zeros(
            batch_size, padded_seq_len, device=device, dtype=torch.float32
        )

        for i, seq in enumerate(sequences):
            length = len(seq.arrival_times)
            mask[i, :length] = 1

        # Other attributes include marks, inter-event times
        if sequences[0].marks is not None:
            marks = pad_sequence([s.marks for s in sequences], max_len=padded_seq_len)
        else:
            marks = None

        return Batch(
            arrival_times=arrival_times,
            inter_times=inter_times,
            t_max=t_max,
            mask=mask,
            marks=marks,
        )

    def get_sequence(self, idx: int) -> Sequence:
        length = int(self.mask[idx].sum(-1))
        arrival_times = self.arrival_times[idx, :length]
        marks = self.marks[idx, :length] if self.marks is not None else None
        t_max = self.t_max[idx].item()
        return Sequence(arrival_times=arrival_times, t_max=t_max, marks=marks)

    def to_list(self) -> List[Sequence]:
        """Convert a batch into a list of variable-length sequences."""
        return [self.get_sequence(idx) for idx in range(self.batch_size)]

    def __repr__(self):
        return f"{self.__class__.__name__}(batch_size={self.batch_size}, seq_len={self.seq_len}, keys={self.keys()})"


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: float = 0,
    max_len: Optional[int] = None,
):
    """Pad a list of variable length Tensors with `padding_value`."""
    dtype = sequences[0].dtype
    device = sequences[0].device
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor
