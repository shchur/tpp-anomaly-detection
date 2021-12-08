from typing import Optional, Union

import numpy as np
import torch

from .utils import DotDict


class Sequence(DotDict):
    """Variable-length sequence of events.

    Args:
        arrival_times: arrival time of each event, shape (seq_len)
        t_max: length of the observed interval.
        marks: categorical marks corresponding to event type, shape (seq_len)
    """

    def __init__(
        self,
        arrival_times: Union[torch.Tensor, np.ndarray, list],
        t_max: Union[float, torch.Tensor],
        marks: Optional[Union[torch.Tensor, np.ndarray, list]] = None,
        metadata: Optional[dict] = None,
    ):
        self.arrival_times = torch.as_tensor(arrival_times)
        self.inter_times = torch.diff(
            self.arrival_times,
            prepend=torch.as_tensor([0.0]),
            append=torch.as_tensor([t_max]),
        ).to(self.arrival_times.dtype)
        self.t_max = torch.as_tensor(float(t_max))

        if marks is not None:
            self.marks = torch.as_tensor(marks, dtype=torch.long)
        else:
            self.marks = None

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        self._validate()

    @property
    def num_events(self):
        return len(self)

    def to_dict(self) -> dict:
        return {k: v for (k, v) in self.items() if v is not None and k != "inter_times"}

    def float(self):
        self.arrival_times = self.arrival_times.float()
        self.inter_times = self.inter_times.float()
        self.t_max = self.t_max.float()
        return self

    def __len__(self):
        return len(self.arrival_times)

    def __add__(self, other: "Sequence") -> "Sequence":
        """Concatenate two event sequences."""
        if not isinstance(other, self.__class__):
            raise ValueError(
                "Sequence can only be concatenated with another Sequence object"
            )

        arrival_times = torch.cat(
            [self.arrival_times, other.arrival_times + self.t_max]
        )
        t_max = self.t_max + other.t_max
        if self.marks is not None and other.marks is not None:
            marks = torch.cat([self.marks, other.marks])
        else:
            marks = None

        if self.metadata:
            metadata = self.metadata
        else:
            metadata = None

        return Sequence(
            arrival_times=arrival_times, t_max=t_max, marks=marks, metadata=metadata
        )

    def _validate(self):
        """Check if all tensors have correct shapes."""
        if self.arrival_times.ndim != 1:
            raise ValueError(
                f"arrival_times must be a 1-d tensor (got {self.arrival_times.ndim}-d)"
            )
        if self.marks is not None and self.marks.shape != (len(self),):
            raise ValueError(
                f"marks must be of shape (seq_len) (got {self.marks.shape})"
            )

    def __repr__(self):
        return f"{self.__class__.__name__}(num_events={self.num_events}, keys={self.keys()})"


def slice_sequence(seq, t_start, t_end):
    """Select the events that happen in (t_start, t_end]."""
    indicator = (seq.arrival_times > t_start) & (seq.arrival_times <= t_end)
    arrival_times = seq.arrival_times[indicator] - t_start
    t_max = min(t_end - t_start, seq.t_max - t_start)
    if seq.marks is not None:
        marks = seq.marks[indicator]
    else:
        marks = None
    return Sequence(arrival_times=arrival_times, t_max=t_max, marks=marks)


def split_into_shorter_sequences(seq, step_size, window_size):
    """Split a long sequence into shorter ones."""
    num_slices = int(np.ceil((seq.t_max - window_size) / step_size))
    subsequences = []
    for i in range(num_slices):
        t_start = i * step_size
        t_end = t_start + window_size
        subsequences.append(slice_sequence(seq, t_start, t_end))
    return subsequences


def dequantize(seq, dt=1.0):
    """Add uniform noise to convert discrete arrival times into continuous.

    If the arrival times are sampled with temporal resolution dt, we add
    Uniform(-0.5*dt, 0.5*dt) noise to convert them into continuous variables.
    """
    times = seq.arrival_times.numpy()
    step = 0.5 * dt
    deq_times = times + np.random.uniform(-step, step, size=len(seq))
    order = np.argsort(deq_times)
    data = {"arrival_times": deq_times[order], "t_max": seq.t_max + step}

    if seq.marks is not None:
        data["marks"] = seq.marks[order]

    return Sequence(**data)
