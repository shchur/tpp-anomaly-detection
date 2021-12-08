from typing import Callable, Iterable, List, Union

import numpy as np
import torch

from .sequence import Sequence


class Compose:
    """Composes several transforms together.

    Args:
        transforms: List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, seq: Sequence) -> Sequence:
        for t in self.transforms:
            seq = t(seq)
        return seq

    def __repr__(self):
        ts = "\n".join([f"    {t}," for t in self.transforms])
        return f"{self.__class__.__name__}([\n{ts}\n])"


class ClampMaxTime:
    """Remove events that happen after new_t_max."""

    def __init__(self, new_t_max: float):
        self.new_t_max = new_t_max

    def __call__(self, seq: Sequence) -> Sequence:
        if self.new_t_max > seq.t_max:
            raise ValueError("new_t_max must be <= old t_max")
        seq.t_max = self.new_t_max
        indicator = seq.arrival_times < self.new_t_max
        for k in seq.keys():
            if isinstance(seq[k], torch.Tensor):
                seq[k] = seq[k][indicator]
        return seq

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Dequantize:
    """Add uniform noise to discrete arrival times to dequantize them."""

    def __init__(self, dt=1.0):
        self.dt = dt

    def __call__(self, seq: Sequence) -> Sequence:
        times = seq.arrival_times.numpy()
        step = 0.5 * self.dt
        deq_times = times + np.random.uniform(-step, step, size=len(seq))
        order = np.argsort(deq_times)
        data = {"arrival_times": deq_times[order], "t_max": seq.t_max + step}

        if seq.marks is not None:
            data["marks"] = seq.marks[order]

        return Sequence(**data)

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SelectMarks:
    """Only keep events that correspond to the given subset of marks."""
    def __init__(self, marks_to_keep: Union[np.ndarray, torch.Tensor]):
        self.marks_to_keep = marks_to_keep

    def __call__(self, seq: Sequence) -> Sequence:
        if seq.marks is None:
            raise ValueError(f"{self} can only be applied to marked sequences.")
        events_to_keep = np.isin(seq.marks.numpy(), self.marks_to_keep)
        return Sequence(
            arrival_times=seq.arrival_times[events_to_keep].as_contiguous(),
            t_max=seq.t_max,
            marks=seq.marks[events_to_keep].as_contiguous(),
        )
