import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

import anomaly_tpp
from anomaly_tpp.distributions import MixtureSameFamily, Weibull


class RecurrentTPP(nn.Module):
    """Marks and times are conditionally independent given the history.

    p(t, x | history) = p(t | history) * p(x | history)

    Args:
        num_marks: Number of marks (classes) in the TPP.
        context_size: Size of the RNN output (i.e. of the context embedding).
        mark_embedding_size: Dimension of the mark embedding.
        num_components: Number of mixture components in the inter-event time distribution.

    """

    def __init__(
        self,
        num_marks: int,
        context_size: int = 64,
        mark_embedding_size: int = 32,
        num_components: int = 8,
    ):
        super().__init__()
        self.context_size = context_size
        self.num_marks = num_marks
        self.num_components = num_components

        num_rnn_input_features = 1
        if num_marks > 1:
            num_rnn_input_features += mark_embedding_size
            self.mark_encoder = nn.Embedding(num_marks, mark_embedding_size)
            self.mark_decoder = nn.Linear(context_size, num_marks)

        self.rnn = nn.GRU(num_rnn_input_features, context_size, batch_first=True)
        self.time_decoder = nn.Linear(context_size, 3 * num_components)

    def get_context(self, batch: anomaly_tpp.data.Batch) -> torch.Tensor:
        """Encode event history into a context vector with an RNN."""
        features = batch.inter_times.clamp_min(1e-8).unsqueeze(-1).log()  # (B, L, 1)
        if self.num_marks > 1:
            mark_emb = self.mark_encoder(batch.marks)  # (B, L, D_mark)
            features = torch.cat([features, mark_emb], dim=-1)  # (B, L, D_mark + 1)
        rnn_output = self.rnn(features)[0]  # (B, L, C)
        context = F.pad(rnn_output[:, :-1, :], [0, 0, 1, 0])  # (B, L, C)
        return context

    def get_output_distribution(self, context: torch.Tensor):
        """Get the inter-event time distribution."""
        params = self.time_decoder(context)  # (B, L, 3 * R)
        params = clamp_preserve_gradients(params, -3.0, 5.0)
        pre_weights, log_scale, log_shape = torch.split(
            params,
            [self.num_components, self.num_components, self.num_components],
            dim=-1,
        )
        log_weights = pre_weights.log_softmax(dim=-1)
        component_dist = Weibull(log_scale=log_scale, log_shape=log_shape)
        mixture_dist = Categorical(logits=log_weights)
        return MixtureSameFamily(mixture_dist, component_dist)

    def nll_loss(self, batch) -> torch.Tensor:
        """Compute negative log-likelihood for each sequence in the batch."""
        context = self.get_context(batch)
        inter_time_dist = self.get_output_distribution(context)

        # Log-intensity for observed events
        log_intensity = inter_time_dist.log_hazard(batch.inter_times)

        if self.num_marks > 1:
            mark_logits = self.mark_decoder(context)
            mark_dist = Categorical(logits=mark_logits)
            log_intensity += mark_dist.log_prob(batch.marks)

        # Compute compensator
        log_survival = inter_time_dist.log_survival(batch.inter_times)
        log_survival = log_survival.masked_fill((batch.inter_times == 0), 0.0)

        log_like = (log_intensity * batch.mask + log_survival).sum(-1)
        return -log_like# / batch.t_max

    def get_compensator_per_mark(self, batch: anomaly_tpp.data.Batch) -> torch.Tensor:
        """Compute compensator at each arrival time for each mark.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            compensator: Integrated intensity for each mark, shape (B, L, K)
        """
        context = self.get_context(batch)  # (B, L, C)
        inter_time_dist = self.get_output_distribution(context)

        log_survival = inter_time_dist.log_survival(batch.inter_times)
        log_survival = log_survival.masked_fill((batch.inter_times == 0), 0.0)

        if self.num_marks > 1:
            mark_logits = self.mark_decoder(context)
            mark_probas = mark_logits.softmax(dim=-1)  # (B, L, K)
        else:
            mark_probas = 1.0

        compensator = (log_survival.neg().unsqueeze(-1) * mark_probas).cumsum(-2)
        return compensator


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region."""
    return x + (x.clamp(min, max) - x).detach()
