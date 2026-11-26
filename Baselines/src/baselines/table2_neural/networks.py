from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..validation import OutcomeType


def _activation(name: str) -> type[nn.Module]:
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    raise ValueError(f"Unsupported neural activation: {name}")


def _mlp(
    input_dim: int,
    *,
    layers: int,
    width: int,
    output_dim: int | None = None,
    activation: str = "elu",
    weight_init_scale: float | None = None,
    hidden_widths: tuple[int, ...] | None = None,
    initialization: str | None = None,
) -> nn.Sequential:
    if layers < 0 or width < 1:
        raise ValueError("neural layer count and width must be positive")
    if weight_init_scale is not None and weight_init_scale <= 0.0:
        raise ValueError("neural weight initialization scale must be positive")
    if initialization not in {None, "jax_stax"}:
        raise ValueError(f"unsupported neural initialization: {initialization}")
    if weight_init_scale is not None and initialization is not None:
        raise ValueError("neural initialization modes are mutually exclusive")
    layer_widths = (
        tuple(int(value) for value in hidden_widths)
        if hidden_widths is not None
        else (width,) * layers
    )
    if any(value < 1 for value in layer_widths):
        raise ValueError("neural hidden widths must be positive")
    activation_class = _activation(activation)
    modules: list[nn.Module] = []
    current = input_dim
    for layer_width in layer_widths:
        linear = nn.Linear(current, layer_width)
        if weight_init_scale is not None:
            nn.init.normal_(
                linear.weight,
                mean=0.0,
                std=weight_init_scale / math.sqrt(current),
            )
            nn.init.zeros_(linear.bias)
        elif initialization == "jax_stax":
            target_std = math.sqrt(2.0 / (current + layer_width))
            pre_truncation_std = target_std / 0.87962566103423978
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=pre_truncation_std,
                a=-2.0 * pre_truncation_std,
                b=2.0 * pre_truncation_std,
            )
            nn.init.normal_(linear.bias, mean=0.0, std=0.01)
        modules.extend([linear, activation_class()])
        current = layer_width
    if output_dim is not None:
        linear = nn.Linear(current, output_dim)
        if weight_init_scale is not None:
            nn.init.normal_(
                linear.weight,
                mean=0.0,
                std=weight_init_scale / math.sqrt(current),
            )
            nn.init.zeros_(linear.bias)
        elif initialization == "jax_stax":
            target_std = math.sqrt(2.0 / (current + output_dim))
            pre_truncation_std = target_std / 0.87962566103423978
            nn.init.trunc_normal_(
                linear.weight,
                mean=0.0,
                std=pre_truncation_std,
                a=-2.0 * pre_truncation_std,
                b=2.0 * pre_truncation_std,
            )
            nn.init.normal_(linear.bias, mean=0.0, std=0.01)
        modules.append(linear)
    return nn.Sequential(*modules)


@dataclass(frozen=True)
class PotentialOutcomeOutput:
    representation: Tensor
    y0: Tensor
    y1: Tensor


@dataclass(frozen=True)
class DragonNetOutput(PotentialOutcomeOutput):
    propensity: Tensor
    epsilon: Tensor


class TARNetNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        representation_layers: int,
        representation_width: int,
        outcome_layers: int,
        outcome_width: int,
        outcome_type: OutcomeType,
        activation: str = "elu",
        weight_init_scale: float | None = None,
        normalize_representation: bool = False,
    ) -> None:
        super().__init__()
        self.outcome_type = outcome_type
        self.normalize_representation = bool(normalize_representation)
        self.representation = _mlp(
            input_dim,
            layers=representation_layers,
            width=representation_width,
            activation=activation,
            weight_init_scale=weight_init_scale,
        )
        representation_dim = representation_width if representation_layers else input_dim
        self.y0_head = _mlp(
            representation_dim,
            layers=outcome_layers,
            width=outcome_width,
            output_dim=1,
            activation=activation,
            weight_init_scale=weight_init_scale,
        )
        self.y1_head = _mlp(
            representation_dim,
            layers=outcome_layers,
            width=outcome_width,
            output_dim=1,
            activation=activation,
            weight_init_scale=weight_init_scale,
        )

    def _outcome(self, head: nn.Module, representation: Tensor) -> Tensor:
        output = head(representation).reshape(-1)
        return torch.sigmoid(output) if self.outcome_type == "binary" else output

    def forward(self, X: Tensor) -> PotentialOutcomeOutput:
        representation = self.representation(X)
        if self.normalize_representation:
            row_norm = torch.sqrt(
                representation.square()
                .sum(dim=1, keepdim=True)
                .clamp_min(1e-10)
            )
            representation = representation / row_norm
        return PotentialOutcomeOutput(
            representation=representation,
            y0=self._outcome(self.y0_head, representation),
            y1=self._outcome(self.y1_head, representation),
        )


class DragonNetNetwork(TARNetNetwork):
    def __init__(
        self,
        *,
        author_initialization: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        first_linear = next(
            module for module in self.y0_head.modules() if isinstance(module, nn.Linear)
        )
        representation_dim = first_linear.in_features
        self.propensity_head = nn.Linear(representation_dim, 1)
        self.epsilon = nn.Parameter(torch.empty(()))
        if author_initialization:
            for layer in self.representation.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=0.05)
                    nn.init.zeros_(layer.bias)
            for module in (self.y0_head, self.y1_head, self.propensity_head):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.zeros_(layer.bias)
            nn.init.normal_(self.epsilon, mean=0.0, std=0.05)
        else:
            nn.init.zeros_(self.epsilon)

    def forward(self, X: Tensor) -> DragonNetOutput:
        base = super().forward(X)
        return DragonNetOutput(
            representation=base.representation,
            y0=base.y0,
            y1=base.y1,
            propensity=torch.sigmoid(
                self.propensity_head(base.representation).reshape(-1)
            ),
            epsilon=self.epsilon,
        )


class OutcomeNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        layers: int,
        width: int,
        outcome_type: OutcomeType,
        activation: str = "elu",
        hidden_widths: tuple[int, ...] | None = None,
        initialization: str | None = None,
    ) -> None:
        super().__init__()
        self.outcome_type = outcome_type
        self.network = _mlp(
            input_dim,
            layers=layers,
            width=width,
            output_dim=1,
            activation=activation,
            hidden_widths=hidden_widths,
            initialization=initialization,
        )

    def forward(self, X: Tensor) -> Tensor:
        output = self.network(X).reshape(-1)
        return torch.sigmoid(output) if self.outcome_type == "binary" else output


class EffectNetwork(OutcomeNetwork):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(outcome_type="continuous", **kwargs)  # type: ignore[arg-type]
