from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from ..validation import OutcomeType


ImbalanceKind = Literal["linear", "rbf", "wasserstein"]


def _groups(representation: Tensor, treatment: Tensor) -> tuple[Tensor, Tensor]:
    treatment_flat = treatment.reshape(-1)
    treated = representation[treatment_flat > 0.5]
    control = representation[treatment_flat <= 0.5]
    if len(treated) == 0 or len(control) == 0:
        raise ValueError("imbalance penalties require both treatment arms")
    return treated, control


def _linear_mmd(representation: Tensor, treatment: Tensor) -> Tensor:
    treated, control = _groups(representation, treatment)
    p = treatment.to(representation.dtype).mean()
    difference = 2.0 * p * treated.mean(0) - 2.0 * (1.0 - p) * control.mean(0)
    return difference.square().sum()


def _rbf_mmd(representation: Tensor, treatment: Tensor, bandwidth: float) -> Tensor:
    if bandwidth <= 0.0:
        raise ValueError("RBF bandwidth must be positive")
    treated, control = _groups(representation, treatment)

    def kernel(left: Tensor, right: Tensor) -> Tensor:
        squared_distance = torch.cdist(left, right).square()
        return torch.exp(-squared_distance / (bandwidth**2))

    return (
        kernel(treated, treated).mean()
        + kernel(control, control).mean()
        - 2.0 * kernel(treated, control).mean()
    ).clamp_min(0.0)


def _sinkhorn_wasserstein(
    representation: Tensor,
    treatment: Tensor,
    *,
    regularization: float,
    iterations: int,
) -> Tensor:
    if regularization <= 0.0:
        raise ValueError("Sinkhorn regularization must be positive")
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")
    treated, control = _groups(representation, treatment)
    cost = torch.cdist(treated, control)
    scale = cost.detach().mean().clamp_min(torch.finfo(cost.dtype).eps)
    kernel = torch.exp(-cost / (regularization * scale)).clamp_min(1e-12)
    a = torch.full(
        (len(treated),), 1.0 / len(treated), dtype=cost.dtype, device=cost.device
    )
    b = torch.full(
        (len(control),), 1.0 / len(control), dtype=cost.dtype, device=cost.device
    )
    u = a
    v = b
    for _ in range(iterations):
        u = a / (kernel @ v).clamp_min(1e-12)
        v = b / (kernel.transpose(0, 1) @ u).clamp_min(1e-12)
    # The original CFRNet default stops gradients through the transport plan.
    transport = (u[:, None] * kernel * v[None, :]).detach()
    return (transport * cost).sum()


def _author_wasserstein(
    representation: Tensor,
    treatment: Tensor,
    *,
    treatment_prevalence: float,
    wasserstein_lambda: float,
    iterations: int,
    backprop_transport: bool,
) -> Tensor:
    if not 0.0 < treatment_prevalence < 1.0:
        raise ValueError("treatment prevalence must be in (0, 1)")
    if wasserstein_lambda <= 0.0:
        raise ValueError("Wasserstein lambda must be positive")
    if iterations < 1:
        raise ValueError("Sinkhorn iterations must be positive")
    treated, control = _groups(representation, treatment)
    squared_distance = (
        -2.0 * treated @ control.transpose(0, 1)
        + treated.square().sum(dim=1, keepdim=True)
        + control.square().sum(dim=1).unsqueeze(0)
    )
    cost = torch.sqrt(torch.abs(squared_distance) + 1e-8)
    delta = cost.max().detach()
    effective_lambda = (float(wasserstein_lambda) / cost.mean()).detach()

    slack_row = delta.expand(1, len(control))
    augmented = torch.cat((cost, slack_row), dim=0)
    slack_column = torch.cat(
        (
            delta.expand(len(treated)),
            torch.zeros((1,), dtype=cost.dtype, device=cost.device),
        )
    ).reshape(-1, 1)
    augmented = torch.cat((augmented, slack_column), dim=1)

    p = float(treatment_prevalence)
    a = torch.cat(
        (
            torch.full(
                (len(treated),), p / len(treated), dtype=cost.dtype, device=cost.device
            ),
            torch.tensor([1.0 - p], dtype=cost.dtype, device=cost.device),
        )
    )
    b = torch.cat(
        (
            torch.full(
                (len(control),),
                (1.0 - p) / len(control),
                dtype=cost.dtype,
                device=cost.device,
            ),
            torch.tensor([p], dtype=cost.dtype, device=cost.device),
        )
    )
    kernel = torch.exp(-effective_lambda * augmented) + 1e-6
    inverse_marginal_kernel = kernel / a[:, None]
    u = a
    for _ in range(iterations):
        column_scaling = b / (u @ kernel)
        u = 1.0 / (inverse_marginal_kernel @ column_scaling)
    v = b / (u @ kernel)
    transport = u[:, None] * (v[None, :] * kernel)
    if not backprop_transport:
        transport = transport.detach()
    return 2.0 * (transport * augmented).sum()


def mmd_penalty(
    representation: Tensor,
    treatment: Tensor,
    *,
    kind: ImbalanceKind = "linear",
    rbf_bandwidth: float = 1.0,
    sinkhorn_regularization: float = 0.1,
    sinkhorn_iterations: int = 10,
    treatment_prevalence: float | None = None,
    author_wasserstein: bool = False,
    wasserstein_lambda: float = 10.0,
    backprop_transport: bool = False,
) -> Tensor:
    """Compute a CFRNet representation discrepancy.

    ``linear`` and ``rbf`` follow the original CFRNet MMD conventions;
    ``wasserstein`` is its entropically regularized transport alternative.
    """

    if representation.ndim != 2:
        raise ValueError("representation must be two-dimensional")
    if treatment.numel() != len(representation):
        raise ValueError("treatment and representation must align")
    if kind == "linear":
        return _linear_mmd(representation, treatment)
    if kind == "rbf":
        return _rbf_mmd(representation, treatment, rbf_bandwidth)
    if kind == "wasserstein":
        if author_wasserstein:
            if treatment_prevalence is None:
                raise ValueError(
                    "author Wasserstein requires treatment prevalence"
                )
            return _author_wasserstein(
                representation,
                treatment,
                treatment_prevalence=treatment_prevalence,
                wasserstein_lambda=wasserstein_lambda,
                iterations=sinkhorn_iterations,
                backprop_transport=backprop_transport,
            )
        return _sinkhorn_wasserstein(
            representation,
            treatment,
            regularization=sinkhorn_regularization,
            iterations=sinkhorn_iterations,
        )
    raise ValueError(f"Unsupported imbalance kind: {kind}")


def factual_outcome_loss(
    y0: Tensor,
    y1: Tensor,
    treatment: Tensor,
    y: Tensor,
    *,
    outcome_type: OutcomeType,
    treatment_prevalence: float | None = None,
) -> Tensor:
    treatment_flat = treatment.reshape(-1).to(y0.dtype)
    factual = treatment_flat * y1.reshape(-1) + (1.0 - treatment_flat) * y0.reshape(-1)
    target = y.reshape(-1).to(factual.dtype)
    sample_weight: Tensor | None = None
    if treatment_prevalence is not None:
        if not 0.0 < treatment_prevalence < 1.0:
            raise ValueError("treatment prevalence must be in (0, 1)")
        p = float(treatment_prevalence)
        sample_weight = treatment_flat / (2.0 * p) + (
            1.0 - treatment_flat
        ) / (2.0 * (1.0 - p))
    if sample_weight is None:
        if outcome_type == "binary":
            return F.binary_cross_entropy(
                factual.clamp(1e-6, 1.0 - 1e-6), target
            )
        return F.mse_loss(factual, target)
    if outcome_type == "binary":
        per_row = F.binary_cross_entropy(
            factual.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
        )
    else:
        per_row = F.mse_loss(factual, target, reduction="none")
    return (sample_weight * per_row).mean()


@dataclass(frozen=True)
class CFRLoss:
    total: Tensor
    factual: Tensor
    imbalance: Tensor
    regularization: Tensor


def cfr_objective(
    y0: Tensor,
    y1: Tensor,
    representation: Tensor,
    treatment: Tensor,
    y: Tensor,
    *,
    outcome_type: OutcomeType,
    alpha: float,
    imbalance_kind: ImbalanceKind,
    regularization: Tensor | None = None,
    regularization_weight: float = 0.0,
    rbf_bandwidth: float = 1.0,
    sinkhorn_regularization: float = 0.1,
    sinkhorn_iterations: int = 10,
    treatment_prevalence: float | None = None,
    imbalance_prevalence: float | None = None,
    author_wasserstein: bool = False,
    wasserstein_lambda: float = 10.0,
    backprop_transport: bool = False,
) -> CFRLoss:
    factual = factual_outcome_loss(
        y0,
        y1,
        treatment,
        y,
        outcome_type=outcome_type,
        treatment_prevalence=treatment_prevalence,
    )
    imbalance = mmd_penalty(
        representation,
        treatment,
        kind=imbalance_kind,
        rbf_bandwidth=rbf_bandwidth,
        sinkhorn_regularization=sinkhorn_regularization,
        sinkhorn_iterations=sinkhorn_iterations,
        treatment_prevalence=imbalance_prevalence,
        author_wasserstein=author_wasserstein,
        wasserstein_lambda=wasserstein_lambda,
        backprop_transport=backprop_transport,
    )
    regularization_term = (
        regularization
        if regularization is not None
        else torch.zeros((), dtype=factual.dtype, device=factual.device)
    )
    return CFRLoss(
        total=factual + float(alpha) * imbalance + float(regularization_weight) * regularization_term,
        factual=factual,
        imbalance=imbalance,
        regularization=regularization_term,
    )


@dataclass(frozen=True)
class DragonNetLoss:
    total: Tensor
    factual: Tensor
    propensity: Tensor
    targeted: Tensor


def dragonnet_objective(
    y0: Tensor,
    y1: Tensor,
    propensity: Tensor,
    epsilon: Tensor,
    treatment: Tensor,
    y: Tensor,
    *,
    outcome_type: OutcomeType,
    targeted_regularization: bool,
    propensity_weight: float,
    targeted_weight: float,
    sum_data_loss: bool = False,
    propensity_clip: tuple[float, float] = (0.01, 0.99),
) -> DragonNetLoss:
    try:
        lower, upper = (float(value) for value in propensity_clip)
    except (TypeError, ValueError) as error:
        raise ValueError("propensity_clip must contain two numeric bounds") from error
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("propensity_clip must lie strictly inside (0, 1)")
    treatment_flat = treatment.reshape(-1).to(y0.dtype)
    target = y.reshape(-1).to(y0.dtype)
    factual = factual_outcome_loss(
        y0, y1, treatment_flat, target, outcome_type=outcome_type
    )
    if sum_data_loss:
        factual = factual * target.numel()
    propensity_smoothed = (propensity.reshape(-1) + 0.001) / 1.002
    if torch.isfinite(propensity_smoothed).all():
        propensity_loss = F.binary_cross_entropy(
            propensity_smoothed.clamp(1e-6, 1.0 - 1e-6),
            treatment_flat,
            reduction="sum" if sum_data_loss else "mean",
        )
    else:
        # Let the trainer's explicit non-finite policy decide whether to fail or
        # restore a previously validated checkpoint. PyTorch BCE otherwise raises
        # before that policy can observe the non-finite objective.
        propensity_loss = propensity_smoothed.sum()
    if targeted_regularization:
        targeted_propensity = (propensity.reshape(-1) + lower) / (
            1.0 + lower + (1.0 - upper)
        )
        clever_covariate = treatment_flat / targeted_propensity - (
            1.0 - treatment_flat
        ) / (1.0 - targeted_propensity)
        factual_prediction = treatment_flat * y1.reshape(-1) + (
            1.0 - treatment_flat
        ) * y0.reshape(-1)
        perturbed = factual_prediction + epsilon.reshape(()) * clever_covariate
        targeted = F.mse_loss(
            perturbed,
            target,
            reduction="sum" if sum_data_loss else "mean",
        )
    else:
        targeted = torch.zeros((), dtype=factual.dtype, device=factual.device)
    return DragonNetLoss(
        total=factual
        + float(propensity_weight) * propensity_loss
        + float(targeted_weight) * targeted,
        factual=factual,
        propensity=propensity_loss,
        targeted=targeted,
    )
