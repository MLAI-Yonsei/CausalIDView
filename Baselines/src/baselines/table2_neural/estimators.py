from __future__ import annotations

from typing import Any, Mapping, cast

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, nn
from typing_extensions import Self

from ..base import BaseCATEEstimator
from ..exceptions import CATEValidationError
from ..meta.common import ConfiguredMetaEstimator
from ..prediction import CATEPrediction, normalize_prediction
from ..validation import OutcomeType, validate_fit_data
from .losses import ImbalanceKind, cfr_objective, factual_outcome_loss
from .networks import PotentialOutcomeOutput, TARNetNetwork
from .trainer import NeuralLossContext, NeuralTrainer, NeuralTrainingResult, predict_tensor


class _BaseNeuralEstimator(ConfiguredMetaEstimator, BaseCATEEstimator):
    objective_name: str

    @property
    def neural_config(self) -> dict[str, Any]:
        return dict(self.config.get("neural", {}))

    def _architecture(self) -> dict[str, Any]:
        neural = self.neural_config
        model_config = dict(self.config.get(self.model_id, {}))
        return {
            "representation_layers": int(
                model_config.get("representation_layers", neural.get("representation_layers", 3))
            ),
            "representation_width": int(
                model_config.get("representation_width", neural.get("representation_width", 200))
            ),
            "outcome_layers": int(
                model_config.get("outcome_layers", neural.get("outcome_layers", 2))
            ),
            "outcome_width": int(
                model_config.get("outcome_width", neural.get("outcome_width", 100))
            ),
            "activation": str(model_config.get("activation", neural.get("activation", "elu"))),
        }

    def _network_kwargs(self, input_dim: int, outcome_type: OutcomeType) -> dict[str, Any]:
        model_config = dict(self.config.get(self.model_id, {}))
        configured_scale = model_config.get("weight_init_scale")
        return {
            "input_dim": input_dim,
            "outcome_type": outcome_type,
            **self._architecture(),
            "weight_init_scale": (
                None if configured_scale is None else float(configured_scale)
            ),
            "normalize_representation": bool(
                model_config.get("normalize_representation", False)
            ),
        }

    def _training_metadata(self, result: NeuralTrainingResult) -> dict[str, Any]:
        return {
            "device": result.device,
            "best_epoch": result.best_epoch,
            "epochs_run": result.epochs_run,
            "best_validation_loss": result.best_validation_loss,
            "termination_reason": result.termination_reason,
            "batch_size": result.batch_size,
            "train_only_scaler_mean": result.scaler.mean_.tolist(),
            "train_only_scaler_scale": result.scaler.scale_.tolist(),
        }

    def _validated_binary_y(self, y: NDArray[np.float64], outcome_type: OutcomeType) -> None:
        if outcome_type == "binary" and not np.isin(y, [0.0, 1.0]).all():
            raise CATEValidationError("binary neural outcomes may contain only 0 and 1")


class _JointPotentialOutcomeEstimator(_BaseNeuralEstimator):
    network_class: type[TARNetNetwork] = TARNetNetwork

    def _loss_function(self, outcome_type: OutcomeType):
        raise NotImplementedError

    @staticmethod
    def _outcome_head_regularization(model: nn.Module) -> Tensor:
        network = cast(TARNetNetwork, model)
        first_parameter = next(network.parameters())
        regularization = torch.zeros(
            (), dtype=first_parameter.dtype, device=first_parameter.device
        )
        for head in (network.y0_head, network.y1_head):
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    regularization = regularization + 0.5 * layer.weight.square().sum()
        return regularization

    def fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> Self:
        X_array, treatment_array, y_array = validate_fit_data(
            X, treatment, y, outcome_type=outcome_type
        )
        self._validated_binary_y(y_array, outcome_type)
        model_config = dict(self.config.get(self.model_id, {}))
        standardize_outcome = bool(
            model_config.get("standardize_outcome", False)
        ) and outcome_type == "continuous"
        self.outcome_standardized_ = standardize_outcome
        self.outcome_mean_ = float(np.mean(y_array)) if standardize_outcome else 0.0
        self.outcome_scale_ = float(np.std(y_array)) if standardize_outcome else 1.0
        if self.outcome_scale_ <= 0.0:
            self.outcome_scale_ = 1.0
        training_y = (
            (y_array - self.outcome_mean_) / self.outcome_scale_
            if standardize_outcome
            else y_array
        )
        configured_batch_size = model_config.get("batch_size")
        trainer = NeuralTrainer(
            self.config,
            batch_size=(
                None if configured_batch_size is None else int(configured_batch_size)
            ),
        )
        trainer.seed_everything()
        network = self.network_class(**self._network_kwargs(X_array.shape[1], outcome_type))
        self.training_result_ = trainer.fit(
            network,
            X_array,
            treatment_array,
            training_y,
            loss_function=self._loss_function(outcome_type),
            stratify=treatment_array,
        )
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self._is_fitted = True
        return self

    def refit_full_context(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
        optimizer_updates: int,
    ) -> Self:
        """Reinitialize and fit every context row for the selected update count."""

        X_array, treatment_array, y_array = validate_fit_data(
            X, treatment, y, outcome_type=outcome_type
        )
        self._validated_binary_y(y_array, outcome_type)
        if not getattr(self, "_is_fitted", False):
            raise CATEValidationError("validation selection must precede full-context refit")
        self.selection_result_ = self.training_result_
        model_config = dict(self.config.get(self.model_id, {}))
        standardize_outcome = bool(
            model_config.get("standardize_outcome", False)
        ) and outcome_type == "continuous"
        self.outcome_standardized_ = standardize_outcome
        self.outcome_mean_ = float(np.mean(y_array)) if standardize_outcome else 0.0
        self.outcome_scale_ = float(np.std(y_array)) if standardize_outcome else 1.0
        if self.outcome_scale_ <= 0.0:
            self.outcome_scale_ = 1.0
        training_y = (
            (y_array - self.outcome_mean_) / self.outcome_scale_
            if standardize_outcome
            else y_array
        )
        configured_batch_size = model_config.get("batch_size")
        trainer = NeuralTrainer(
            self.config,
            batch_size=(
                None if configured_batch_size is None else int(configured_batch_size)
            ),
        )
        trainer.seed_everything()
        network = self.network_class(**self._network_kwargs(X_array.shape[1], outcome_type))
        self.training_result_ = trainer.refit(
            network,
            X_array,
            treatment_array,
            training_y,
            loss_function=self._loss_function(outcome_type),
            optimizer_updates=int(optimizer_updates),
            stratify=treatment_array,
        )
        self.selected_optimizer_updates_ = int(optimizer_updates)
        self.n_features_in_ = X_array.shape[1]
        return self

    def predict(
        self,
        X: NDArray[np.float64],
        *,
        return_po: bool = False,
        return_interval: bool = False,
        alpha: float = 0.05,
    ) -> CATEPrediction:
        X_array = self._validated_predict_X(X, alpha=alpha)
        if return_interval:
            raise CATEValidationError(f"{self.model_id} does not provide intervals")
        output = cast(PotentialOutcomeOutput, predict_tensor(self.training_result_, X_array))
        mu0 = output.y0.detach().cpu().numpy().astype(np.float64)
        mu1 = output.y1.detach().cpu().numpy().astype(np.float64)
        if self.outcome_standardized_:
            mu0 = self.outcome_mean_ + self.outcome_scale_ * mu0
            mu1 = self.outcome_mean_ + self.outcome_scale_ * mu1
        cate = mu1 - mu0
        return normalize_prediction(
            CATEPrediction(
                cate=cate,
                mu0=mu0 if return_po else None,
                mu1=mu1 if return_po else None,
                metadata=self.get_metadata(),
            ),
            len(X_array),
        )

    def get_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "seed": self.seed,
            "backend": "pytorch",
            "objective": self.objective_name,
            "architecture": self._architecture(),
            "binary_outcome_handling": "native probability risk difference",
            "training_config": self.neural_config,
        }
        if getattr(self, "_is_fitted", False):
            metadata.update(self._training_metadata(self.training_result_))
            metadata.update(
                outcome_standardized=self.outcome_standardized_,
                outcome_mean=self.outcome_mean_,
                outcome_scale=self.outcome_scale_,
            )
            if hasattr(self, "selection_result_"):
                metadata.update(
                    full_context_refit=True,
                    selected_optimizer_updates=self.selected_optimizer_updates_,
                    selection_training_indices=list(
                        self.selection_result_.training_indices
                    ),
                    selection_validation_indices=list(
                        self.selection_result_.validation_indices
                    ),
                    final_training_indices=list(self.training_result_.training_indices),
                    selection_scaler_mean=self.selection_result_.scaler.mean_.tolist(),
                    selection_scaler_scale=self.selection_result_.scaler.scale_.tolist(),
                    final_scaler_mean=self.training_result_.scaler.mean_.tolist(),
                    final_scaler_scale=self.training_result_.scaler.scale_.tolist(),
                )
        return metadata


class TARNetEstimator(_JointPotentialOutcomeEstimator):
    model_id = "tarnet"
    objective_name = "factual outcome loss"

    def _loss_function(self, outcome_type: OutcomeType):
        tarnet = dict(self.config.get("tarnet", {}))
        reweight_sample = bool(tarnet.get("reweight_sample", False))
        regularization_weight = float(
            tarnet.get("outcome_head_regularization", 0.0)
        )

        def loss(
            model: nn.Module,
            X: Tensor,
            treatment: Tensor,
            y: Tensor,
            context: NeuralLossContext,
        ) -> Tensor:
            output = cast(PotentialOutcomeOutput, model(X))
            factual = factual_outcome_loss(
                output.y0,
                output.y1,
                treatment,
                y,
                outcome_type=outcome_type,
                treatment_prevalence=(
                    context.treatment_prevalence if reweight_sample else None
                ),
            )
            return factual + regularization_weight * self._outcome_head_regularization(
                model
            )

        return loss

    def get_metadata(self) -> Mapping[str, Any]:
        return {**super().get_metadata(), "balance_penalty": 0.0}


class CFRNetEstimator(_JointPotentialOutcomeEstimator):
    model_id = "cfrnet"
    objective_name = "factual outcome loss + representation imbalance"

    def _loss_function(self, outcome_type: OutcomeType):
        cfr = dict(self.config.get("cfrnet", {}))
        alpha = float(cfr.get("alpha", 1.0))
        kind = cast(ImbalanceKind, str(cfr.get("imbalance", "wasserstein")))
        reweight_sample = bool(cfr.get("reweight_sample", False))
        use_prevalence_for_imbalance = bool(
            cfr.get("use_treatment_prevalence_for_imbalance", True)
        )

        def loss(
            model: nn.Module,
            X: Tensor,
            treatment: Tensor,
            y: Tensor,
            context: NeuralLossContext,
        ) -> Tensor:
            output = cast(PotentialOutcomeOutput, model(X))
            return cfr_objective(
                output.y0,
                output.y1,
                output.representation,
                treatment,
                y,
                outcome_type=outcome_type,
                alpha=alpha,
                imbalance_kind=kind,
                rbf_bandwidth=float(cfr.get("rbf_bandwidth", 1.0)),
                sinkhorn_regularization=float(cfr.get("sinkhorn_regularization", 0.1)),
                sinkhorn_iterations=int(cfr.get("sinkhorn_iterations", 10)),
                treatment_prevalence=(
                    context.treatment_prevalence if reweight_sample else None
                ),
                imbalance_prevalence=(
                    context.treatment_prevalence
                    if use_prevalence_for_imbalance
                    else 0.5
                ),
                author_wasserstein=bool(cfr.get("author_wasserstein", False)),
                wasserstein_lambda=float(cfr.get("wasserstein_lambda", 10.0)),
                backprop_transport=bool(cfr.get("backprop_transport", False)),
                regularization=self._outcome_head_regularization(model),
                regularization_weight=float(
                    cfr.get("outcome_head_regularization", 0.0)
                ),
            ).total

        return loss

    def get_metadata(self) -> Mapping[str, Any]:
        cfr = dict(self.config.get("cfrnet", {}))
        return {
            **super().get_metadata(),
            "imbalance": str(cfr.get("imbalance", "wasserstein")),
            "alpha": float(cfr.get("alpha", 1.0)),
        }
