import anomaly_tpp
import torch


class StandardPoissonProcess:
    """Computes the compensator of the standard Poisson process (identity function).

    This class is used in the experiments from Section 6.1.
    """
    def __init__(self, num_marks=1):
        self.num_marks = num_marks

    def get_compensator_per_mark(self, batch: anomaly_tpp.data.Batch) -> torch.Tensor:
        t = batch.arrival_times
        return t.unsqueeze(-1).expand([*t.shape, self.num_marks])  # (B, L, K)
