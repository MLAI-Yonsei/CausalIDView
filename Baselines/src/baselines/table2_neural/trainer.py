from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import math
import os
import random
from typing import Callable, Mapping
import warnings

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, train_test_split
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..exceptions import CATEValidationError, DependencyUnavailableError
from ..validation import OutcomeType


@dataclass(frozen=True)
class NeuralLossContext:
    treatment_prevalence: float


LossFunction = Callable[..., Tensor]


@dataclass(frozen=True)
class TorchStandardizer:
    mean_: NDArray[np.float64]
    scale_: NDArray[np.float64]

    @classmethod
    def fit(cls, X: NDArray[np.float64]) -> "TorchStandardizer":
        X_array = np.asarray(X, dtype=np.float64)
        if X_array.ndim != 2 or len(X_array) == 0:
            raise CATEValidationError("standardizer training X must be nonempty and 2D")
        mean = X_array.mean(axis=0)
        scale = X_array.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0)
        return cls(mean_=mean, scale_=scale)

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: nn.Module
    scaler: TorchStandardizer
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    best_epoch: int
    epochs_run: int
    best_validation_loss: float
    training_history: tuple[float, ...]
    validation_history: tuple[float, ...]
    device: str
    batch_size: int
    optimizer_updates: int = 0
    selected_optimizer_updates: int = 0
    termination_reason: str | None = None
    seed: int = 0


class NeuralTrainer:
    def __init__(
        self,
        config: Mapping[str, object],
        *,
        seed_offset: int = 0,
        batch_size: int | None = None,
    ) -> None:
        self.config = dict(config)
        self.neural = dict(self.config.get("neural", {}))  # type: ignore[arg-type]
        self.seed = int(self.config.get("seed", 0)) + int(seed_offset)
        self.max_epochs = int(self.neural.get("max_epochs", 1000))
        self.patience = int(self.neural.get("patience", 50))
        self.validation_fraction = float(self.neural.get("validation_fraction", 0.2))
        self.learning_rate = float(self.neural.get("learning_rate", 1e-3))
        self.weight_decay = float(self.neural.get("weight_decay", 1e-4))
        configured_steps = self.neural.get("training_steps")
        self.training_steps = (
            None if configured_steps is None else int(configured_steps)
        )
        self.validation_interval = int(self.neural.get("validation_interval", 100))
        self.learning_rate_decay = float(
            self.neural.get("learning_rate_decay", 1.0)
        )
        self.learning_rate_decay_interval = int(
            self.neural.get("learning_rate_decay_interval", self.validation_interval)
        )
        self.non_finite_policy = str(self.neural.get("non_finite_policy", "error"))
        self.restore_best_weights = bool(
            self.neural.get("restore_best_weights", True)
        )
        self.epoch_batching = str(
            self.neural.get("epoch_batching", "balanced")
        )
        self.split_strategy = str(self.neural.get("split_strategy", "default"))
        self.minimum_optimizer_updates = int(
            self.neural.get("minimum_optimizer_updates", 0)
        )
        self.kernel_l2 = float(self.neural.get("kernel_l2", 0.0))
        configured_batch_size = (
            batch_size if batch_size is not None else self.neural.get("batch_size")
        )
        self.batch_size = (
            None if configured_batch_size is None else int(configured_batch_size)
        )
        self.device = self._resolve_device()
        if self.max_epochs < 1 or self.patience < 1:
            raise CATEValidationError("neural max_epochs and patience must be positive")
        if not 0.0 < self.validation_fraction < 0.5:
            raise CATEValidationError("neural validation_fraction must be in (0, 0.5)")
        if self.batch_size is not None and self.batch_size < 1:
            raise CATEValidationError("neural batch_size must be positive")
        if self.training_steps is not None and self.training_steps < 1:
            raise CATEValidationError("neural training_steps must be positive")
        if self.validation_interval < 1:
            raise CATEValidationError("neural validation_interval must be positive")
        if not 0.0 < self.learning_rate_decay <= 1.0:
            raise CATEValidationError("neural learning_rate_decay must be in (0, 1]")
        if self.learning_rate_decay_interval < 1:
            raise CATEValidationError(
                "neural learning_rate_decay_interval must be positive"
            )
        if self.non_finite_policy not in {"error", "restore_best_checkpoint"}:
            raise CATEValidationError(
                "neural non_finite_policy must be error or restore_best_checkpoint"
            )
        if self.epoch_batching not in {"balanced", "keras", "catenets_legacy"}:
            raise CATEValidationError(
                "neural epoch_batching must be balanced, keras, or catenets_legacy"
            )
        if self.split_strategy not in {"default", "catenets"}:
            raise CATEValidationError(
                "neural split_strategy must be default or catenets"
            )
        if self.minimum_optimizer_updates < 0:
            raise CATEValidationError(
                "neural minimum_optimizer_updates must be nonnegative"
            )
        if self.kernel_l2 < 0.0:
            raise CATEValidationError("neural kernel_l2 must be nonnegative")

    def _resolve_device(self) -> torch.device:
        requested = str(
            self.neural.get("device", self.config.get("gpu_device", "cpu"))
        )
        if requested == "cpu":
            os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
            return torch.device("cpu")
        if requested in {"cuda", "cuda:0", "auto"}:
            available = torch.cuda.is_available()
            if requested in {"cuda", "cuda:0"} and not available:
                raise DependencyUnavailableError("CUDA was requested but is unavailable")
            if requested == "auto":
                return torch.device("cuda" if available else "cpu")
            return torch.device(requested)
        raise CATEValidationError(
            "neural device must be cpu, cuda, cuda:0, or auto; isolated GPU "
            "workers must use logical cuda:0"
        )

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(int(self.config.get("thread_count", 1)))
        torch.use_deterministic_algorithms(True)

    def _split(
        self,
        X: NDArray[np.float64],
        stratify: NDArray[np.int64] | None,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        n_rows = len(X)
        if n_rows < 3:
            raise CATEValidationError("neural training requires at least three rows")
        predefined = self.neural.get("predefined_validation_indices")
        if predefined is not None:
            try:
                validation_indices = np.asarray(predefined, dtype=np.int64)
            except (TypeError, ValueError) as error:
                raise CATEValidationError(
                    "predefined validation indices must be integers"
                ) from error
            if (
                validation_indices.ndim != 1
                or len(validation_indices) == 0
                or len(np.unique(validation_indices)) != len(validation_indices)
                or np.any(validation_indices < 0)
                or np.any(validation_indices >= n_rows)
            ):
                raise CATEValidationError(
                    "predefined validation indices must be unique and in range"
                )
            validation_set = set(int(value) for value in validation_indices)
            train_indices = np.asarray(
                [index for index in range(n_rows) if index not in validation_set],
                dtype=np.int64,
            )
            if len(train_indices) < 2:
                raise CATEValidationError(
                    "predefined validation split requires at least two training rows"
                )
            if stratify is not None and set(np.unique(stratify[train_indices])) != {0, 1}:
                raise CATEValidationError(
                    "predefined neural training split requires both treatment arms"
                )
            return train_indices, validation_indices
        validation_size = max(1, int(math.ceil(n_rows * self.validation_fraction)))
        validation_size = min(validation_size, n_rows - 2)
        if self.split_strategy == "catenets":
            indices = np.arange(n_rows, dtype=np.int64)
            train_indices, validation_indices = train_test_split(
                indices,
                test_size=self.validation_fraction,
                random_state=self.seed,
                shuffle=True,
                stratify=stratify,
            )
            return (
                np.asarray(train_indices, dtype=np.int64),
                np.asarray(validation_indices, dtype=np.int64),
            )
        if stratify is not None:
            labels, counts = np.unique(stratify, return_counts=True)
            if len(labels) > 1 and counts.min() >= 2:
                validation_size = max(validation_size, len(labels))
                validation_size = min(validation_size, n_rows - len(labels))
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=validation_size,
                    random_state=self.seed,
                )
                train_indices, validation_indices = next(
                    splitter.split(X, stratify)
                )
                return train_indices, validation_indices
        splitter = ShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=self.seed,
        )
        return next(splitter.split(X))

    @staticmethod
    def _tensor(value: NDArray[np.float64] | NDArray[np.int64], device: torch.device) -> Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device)

    def _epoch_batches(
        self,
        train_indices: NDArray[np.int64],
        treatment: NDArray[np.int64],
        *,
        epoch: int,
        require_mixed_arms: bool,
    ) -> list[NDArray[np.int64]]:
        batch_size = min(self.batch_size or len(train_indices), len(train_indices))
        rng = np.random.default_rng(self.seed + epoch)
        shuffled = rng.permutation(train_indices).astype(np.int64, copy=False)
        if self.epoch_batching == "catenets_legacy":
            shuffled = np.asarray(train_indices, dtype=np.int64).copy()
            legacy_rng = np.random.RandomState(self.seed)
            for _ in range(epoch + 1):
                legacy_rng.shuffle(shuffled)
            n_batches = (
                int(np.round(len(shuffled) / batch_size))
                if batch_size < len(shuffled)
                else 1
            )
            return [
                shuffled[
                    batch * batch_size : min(
                        (batch + 1) * batch_size, len(shuffled) - 1
                    )
                ].astype(np.int64, copy=True)
                for batch in range(n_batches)
            ]
        if self.epoch_batching == "keras":
            return [
                shuffled[start : start + batch_size].astype(np.int64, copy=True)
                for start in range(0, len(shuffled), batch_size)
            ]
        if require_mixed_arms and batch_size < 2:
            raise CATEValidationError(
                "treatment-stratified neural batches require batch_size >= 2"
            )
        n_batches = int(math.ceil(len(shuffled) / batch_size))
        batches = [batch.astype(np.int64, copy=True) for batch in np.array_split(shuffled, n_batches)]
        if not require_mixed_arms:
            return batches
        arm_rows = {
            arm: train_indices[treatment[train_indices] == arm]
            for arm in (0, 1)
        }
        for batch in batches:
            present = set(int(value) for value in treatment[batch])
            if present == {0, 1}:
                continue
            missing_arm = 1 if present == {0} else 0
            replacement_pool = arm_rows[missing_arm]
            replacement = int(replacement_pool[rng.integers(len(replacement_pool))])
            batch[-1] = replacement
        return batches

    def _step_batch(
        self,
        train_indices: NDArray[np.int64],
        treatment: NDArray[np.int64],
        *,
        step: int,
        require_mixed_arms: bool,
    ) -> NDArray[np.int64]:
        batch_size = min(self.batch_size or len(train_indices), len(train_indices))
        if require_mixed_arms and batch_size < 2:
            raise CATEValidationError(
                "treatment-stratified neural batches require batch_size >= 2"
            )
        rng = np.random.default_rng(self.seed + step)
        batch = rng.choice(train_indices, size=batch_size, replace=False).astype(
            np.int64, copy=False
        )
        if require_mixed_arms and set(int(value) for value in treatment[batch]) != {0, 1}:
            present = set(int(value) for value in treatment[batch])
            missing_arm = 1 if present == {0} else 0
            replacement_pool = train_indices[treatment[train_indices] == missing_arm]
            batch[-1] = int(replacement_pool[rng.integers(len(replacement_pool))])
        return batch

    def _optimizer(
        self,
        model: nn.Module,
        phase: Mapping[str, object],
    ) -> torch.optim.Optimizer:
        name = str(phase.get("optimizer", "adam")).lower()
        learning_rate = float(phase.get("learning_rate", self.learning_rate))
        weight_decay = float(phase.get("weight_decay", self.weight_decay))
        if name == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        if name == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=float(phase.get("momentum", 0.0)),
                nesterov=bool(phase.get("nesterov", False)),
            )
        raise CATEValidationError(f"unsupported neural optimizer phase: {name}")

    @staticmethod
    def linear_kernel_l2(model: nn.Module, penalty: float) -> Tensor:
        first_parameter = next(model.parameters())
        regularization = torch.zeros(
            (), dtype=first_parameter.dtype, device=first_parameter.device
        )
        for module in model.modules():
            if isinstance(module, nn.Linear):
                regularization = (
                    regularization
                    + 0.5 * penalty * module.weight.square().sum()
                )
        return regularization

    @staticmethod
    def _plateau_scheduler(
        optimizer: torch.optim.Optimizer,
        phase: Mapping[str, object],
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
        plateau_patience = int(phase.get("plateau_patience", 0))
        if plateau_patience <= 0:
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(phase.get("plateau_mode", "min")),
            factor=float(phase.get("plateau_factor", 0.5)),
            patience=plateau_patience,
            threshold=float(phase.get("plateau_threshold", 1e-4)),
            threshold_mode=str(phase.get("plateau_threshold_mode", "rel")),
            cooldown=int(phase.get("plateau_cooldown", 0)),
            min_lr=float(phase.get("plateau_min_lr", 0.0)),
        )

    def fit(
        self,
        model: nn.Module,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        loss_function: LossFunction,
        stratify: NDArray[np.int64] | None,
    ) -> NeuralTrainingResult:
        self.seed_everything()
        train_indices, validation_indices = self._split(X, stratify)
        loss_context = NeuralLossContext(
            treatment_prevalence=float(np.mean(treatment[train_indices]))
        )
        loss_parameters = tuple(inspect.signature(loss_function).parameters.values())
        accepts_context = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in loss_parameters
        ) or len(
            [
                parameter
                for parameter in loss_parameters
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
        ) >= 5

        def evaluate_loss(
            fitted: nn.Module,
            X_batch: Tensor,
            treatment_batch: Tensor,
            y_batch: Tensor,
        ) -> Tensor:
            if accepts_context:
                return loss_function(
                    fitted, X_batch, treatment_batch, y_batch, loss_context
                )
            return loss_function(fitted, X_batch, treatment_batch, y_batch)
        if bool(self.neural.get("standardize_features", True)):
            scaler = TorchStandardizer.fit(X[train_indices])
        else:
            scaler = TorchStandardizer(
                mean_=np.zeros(X.shape[1], dtype=np.float64),
                scale_=np.ones(X.shape[1], dtype=np.float64),
            )
        X_scaled = scaler.transform(X)
        X_tensor = self._tensor(X_scaled, self.device)
        treatment_tensor = self._tensor(treatment, self.device)
        y_tensor = self._tensor(y, self.device)
        model = model.to(self.device)
        validation_index_tensor = torch.as_tensor(validation_indices, device=self.device)
        best_state: dict[str, Tensor] | None = None
        best_validation = math.inf
        best_epoch = -1
        best_optimizer_updates = 0
        train_history: list[float] = []
        validation_history: list[float] = []
        require_mixed_arms = (
            stratify is not None and set(np.unique(treatment[train_indices])) == {0, 1}
        )
        effective_batch_size = min(self.batch_size or len(train_indices), len(train_indices))
        if self.training_steps is not None:
            if self.neural.get("optimizer_phases") is not None:
                raise CATEValidationError(
                    "training_steps and optimizer_phases are mutually exclusive"
                )
            optimizer = self._optimizer(model, {"optimizer": "adam"})
            for step in range(self.training_steps):
                model.train()
                batch_indices = self._step_batch(
                    train_indices,
                    treatment,
                    step=step,
                    require_mixed_arms=require_mixed_arms,
                )
                batch_index_tensor = torch.as_tensor(batch_indices, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                train_loss = evaluate_loss(
                    model,
                    X_tensor[batch_index_tensor],
                    treatment_tensor[batch_index_tensor],
                    y_tensor[batch_index_tensor],
                )
                if not torch.isfinite(train_loss):
                    raise CATEValidationError("neural training loss became non-finite")
                with warnings.catch_warnings():
                    if self.device.type == "cpu":
                        warnings.filterwarnings(
                            "ignore", message="CUDA initialization:.*", category=UserWarning
                        )
                        warnings.filterwarnings(
                            "ignore", message="Can't initialize NVML", category=UserWarning
                        )
                    train_loss.backward()
                    optimizer.step()
                train_history.append(float(train_loss.detach().cpu()))
                if (
                    self.learning_rate_decay != 1.0
                    and (step + 1) % self.learning_rate_decay_interval == 0
                ):
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] *= self.learning_rate_decay

                should_validate = (
                    (step + 1) % self.validation_interval == 0
                    or step + 1 == self.training_steps
                )
                if not should_validate:
                    continue
                model.eval()
                with torch.no_grad():
                    validation_loss = evaluate_loss(
                        model,
                        X_tensor[validation_index_tensor],
                        treatment_tensor[validation_index_tensor],
                        y_tensor[validation_index_tensor],
                    )
                validation_value = float(validation_loss.detach().cpu())
                validation_history.append(validation_value)
                if validation_value < best_validation:
                    best_validation = validation_value
                    best_epoch = step
                    best_optimizer_updates = step + 1
                    best_state = deepcopy(model.state_dict())
            if best_state is None:
                raise CATEValidationError("neural training did not produce a checkpoint")
            model.load_state_dict(best_state)
            model.eval()
            return NeuralTrainingResult(
                model=model,
                scaler=scaler,
                training_indices=tuple(int(index) for index in train_indices),
                validation_indices=tuple(int(index) for index in validation_indices),
                best_epoch=best_epoch,
                epochs_run=self.training_steps,
                best_validation_loss=best_validation,
                training_history=tuple(train_history),
                validation_history=tuple(validation_history),
                device=str(self.device),
                batch_size=effective_batch_size,
                optimizer_updates=self.training_steps,
                selected_optimizer_updates=best_optimizer_updates,
                seed=self.seed,
            )
        configured_phases = self.neural.get("optimizer_phases")
        if configured_phases is None:
            phases: list[Mapping[str, object]] = [
                {
                    "optimizer": "adam",
                    "epochs": self.max_epochs,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "patience": self.patience,
                }
            ]
        elif not isinstance(configured_phases, list) or not configured_phases:
            raise CATEValidationError("neural optimizer_phases must be a non-empty list")
        else:
            if any(not isinstance(phase, Mapping) for phase in configured_phases):
                raise CATEValidationError("each neural optimizer phase must be a mapping")
            phases = list(configured_phases)

        global_epoch = 0
        optimizer_updates = 0
        termination_reason: str | None = None
        terminate_training = False
        for phase in phases:
            phase_epochs = int(phase.get("epochs", 0))
            phase_patience = int(phase.get("patience", self.patience))
            if phase_epochs < 1 or phase_patience < 1:
                raise CATEValidationError(
                    "optimizer phase epochs and patience must be positive"
                )
            optimizer = self._optimizer(model, phase)
            scheduler = self._plateau_scheduler(optimizer, phase)
            phase_best = math.inf
            stale_epochs = 0
            for _ in range(phase_epochs):
                model.train()
                epoch_loss_sum = 0.0
                epoch_rows = 0
                epoch_batches = self._epoch_batches(
                    train_indices,
                    treatment,
                    epoch=global_epoch,
                    require_mixed_arms=require_mixed_arms,
                )
                for batch_indices in epoch_batches:
                    batch_index_tensor = torch.as_tensor(batch_indices, device=self.device)
                    optimizer.zero_grad(set_to_none=True)
                    train_loss = evaluate_loss(
                        model,
                        X_tensor[batch_index_tensor],
                        treatment_tensor[batch_index_tensor],
                        y_tensor[batch_index_tensor],
                    )
                    if not torch.isfinite(train_loss):
                        if (
                            self.non_finite_policy == "restore_best_checkpoint"
                            and best_state is not None
                        ):
                            termination_reason = "non_finite_training_loss"
                            terminate_training = True
                            break
                        raise CATEValidationError("neural training loss became non-finite")
                    with warnings.catch_warnings():
                        if self.device.type == "cpu":
                            warnings.filterwarnings(
                                "ignore", message="CUDA initialization:.*", category=UserWarning
                            )
                            warnings.filterwarnings(
                                "ignore", message="Can't initialize NVML", category=UserWarning
                            )
                        train_loss.backward()
                        optimizer.step()
                    optimizer_updates += 1
                    epoch_loss_sum += float(train_loss.detach().cpu()) * len(batch_indices)
                    epoch_rows += len(batch_indices)
                if terminate_training:
                    break
                model.eval()
                with torch.no_grad():
                    validation_loss = evaluate_loss(
                        model,
                        X_tensor[validation_index_tensor],
                        treatment_tensor[validation_index_tensor],
                        y_tensor[validation_index_tensor],
                    )
                train_value = epoch_loss_sum / epoch_rows
                validation_value = float(validation_loss.detach().cpu())
                if not math.isfinite(validation_value):
                    if (
                        self.non_finite_policy == "restore_best_checkpoint"
                        and best_state is not None
                    ):
                        termination_reason = "non_finite_validation_loss"
                        terminate_training = True
                        break
                    raise CATEValidationError(
                        "neural validation loss became non-finite"
                    )
                train_history.append(train_value)
                validation_history.append(validation_value)
                if scheduler is not None:
                    scheduler.step(train_value)
                catenets_eligible = (
                    self.epoch_batching != "catenets_legacy"
                    or optimizer_updates > self.minimum_optimizer_updates
                )
                if catenets_eligible:
                    if validation_value < best_validation:
                        best_validation = validation_value
                        best_epoch = global_epoch
                        best_optimizer_updates = optimizer_updates
                        best_state = deepcopy(model.state_dict())
                    if validation_value < phase_best:
                        phase_best = validation_value
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                global_epoch += 1
                should_stop = (
                    stale_epochs > phase_patience
                    if self.epoch_batching == "catenets_legacy"
                    else stale_epochs >= phase_patience
                )
                if catenets_eligible and should_stop:
                    termination_reason = "early_stopping"
                    break
            if terminate_training:
                break
        if best_state is None:
            if self.epoch_batching != "catenets_legacy" or not validation_history:
                raise CATEValidationError("neural training did not produce a checkpoint")
            best_state = deepcopy(model.state_dict())
            best_epoch = global_epoch - 1
            best_validation = validation_history[-1]
            best_optimizer_updates = optimizer_updates
        if self.restore_best_weights or (
            termination_reason is not None
            and self.epoch_batching != "catenets_legacy"
        ):
            model.load_state_dict(best_state)
        model.eval()
        return NeuralTrainingResult(
            model=model,
            scaler=scaler,
            training_indices=tuple(int(index) for index in train_indices),
            validation_indices=tuple(int(index) for index in validation_indices),
            best_epoch=best_epoch,
            epochs_run=len(train_history),
            best_validation_loss=best_validation,
            training_history=tuple(train_history),
            validation_history=tuple(validation_history),
            device=str(self.device),
            batch_size=effective_batch_size,
            optimizer_updates=optimizer_updates,
            selected_optimizer_updates=best_optimizer_updates,
            termination_reason=termination_reason,
            seed=self.seed,
        )

    def refit(
        self,
        model: nn.Module,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        loss_function: LossFunction,
        optimizer_updates: int,
        stratify: NDArray[np.int64] | None,
    ) -> NeuralTrainingResult:
        """Freshly fit on every supplied row for an exact update budget.

        The update budget must have been selected without query outcomes. The
        fixed failure-search profile uses one Adam phase, so refitting rejects
        multi-phase schedules instead of silently changing their semantics.
        """

        if optimizer_updates < 1:
            raise CATEValidationError("refit optimizer_updates must be positive")
        if self.neural.get("optimizer_phases") is not None:
            raise CATEValidationError(
                "exact full-context refit does not support optimizer_phases"
            )
        X_array = np.asarray(X, dtype=np.float64)
        treatment_array = np.asarray(treatment, dtype=np.int64)
        y_array = np.asarray(y, dtype=np.float64)
        if X_array.ndim != 2 or len(X_array) < 2:
            raise CATEValidationError("neural refit X must be nonempty and two-dimensional")
        if treatment_array.shape != (len(X_array),) or y_array.shape != (len(X_array),):
            raise CATEValidationError("neural refit arrays must have matching rows")
        if not (
            np.isfinite(X_array).all()
            and np.isfinite(treatment_array).all()
            and np.isfinite(y_array).all()
        ):
            raise CATEValidationError("neural refit arrays must be finite")
        if stratify is not None and set(np.unique(treatment_array)) != {0, 1}:
            raise CATEValidationError("neural refit requires both treatment arms")

        self.seed_everything()
        all_indices = np.arange(len(X_array), dtype=np.int64)
        loss_context = NeuralLossContext(
            treatment_prevalence=float(np.mean(treatment_array))
        )
        loss_parameters = tuple(inspect.signature(loss_function).parameters.values())
        accepts_context = any(
            parameter.kind is inspect.Parameter.VAR_POSITIONAL
            for parameter in loss_parameters
        ) or len(
            [
                parameter
                for parameter in loss_parameters
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
        ) >= 5

        if bool(self.neural.get("standardize_features", True)):
            scaler = TorchStandardizer.fit(X_array)
        else:
            scaler = TorchStandardizer(
                mean_=np.zeros(X_array.shape[1], dtype=np.float64),
                scale_=np.ones(X_array.shape[1], dtype=np.float64),
            )
        X_tensor = self._tensor(scaler.transform(X_array), self.device)
        treatment_tensor = self._tensor(treatment_array, self.device)
        y_tensor = self._tensor(y_array, self.device)
        fitted = model.to(self.device)
        optimizer = self._optimizer(fitted, {"optimizer": "adam"})
        history: list[float] = []
        require_mixed_arms = stratify is not None
        for step in range(optimizer_updates):
            fitted.train()
            batch_indices = self._step_batch(
                all_indices,
                treatment_array,
                step=step,
                require_mixed_arms=require_mixed_arms,
            )
            batch_index_tensor = torch.as_tensor(batch_indices, device=self.device)
            optimizer.zero_grad(set_to_none=True)
            if accepts_context:
                loss = loss_function(
                    fitted,
                    X_tensor[batch_index_tensor],
                    treatment_tensor[batch_index_tensor],
                    y_tensor[batch_index_tensor],
                    loss_context,
                )
            else:
                loss = loss_function(
                    fitted,
                    X_tensor[batch_index_tensor],
                    treatment_tensor[batch_index_tensor],
                    y_tensor[batch_index_tensor],
                )
            if not torch.isfinite(loss):
                raise CATEValidationError("neural refit loss became non-finite")
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach().cpu()))
            if (
                self.learning_rate_decay != 1.0
                and (step + 1) % self.learning_rate_decay_interval == 0
            ):
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] *= self.learning_rate_decay
        fitted.eval()
        effective_batch_size = min(self.batch_size or len(all_indices), len(all_indices))
        return NeuralTrainingResult(
            model=fitted,
            scaler=scaler,
            training_indices=tuple(int(index) for index in all_indices),
            validation_indices=(),
            best_epoch=-1,
            epochs_run=0,
            best_validation_loss=0.0,
            training_history=tuple(history),
            validation_history=(),
            device=str(self.device),
            batch_size=effective_batch_size,
            optimizer_updates=optimizer_updates,
            selected_optimizer_updates=optimizer_updates,
            termination_reason="full_context_refit",
            seed=self.seed,
        )

    def fit_supervised(
        self,
        model: nn.Module,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> NeuralTrainingResult:
        treatment = np.zeros(len(X), dtype=np.int64)

        def supervised_loss(
            fitted: nn.Module,
            X_batch: Tensor,
            treatment_batch: Tensor,
            y_batch: Tensor,
            context: NeuralLossContext,
        ) -> Tensor:
            prediction = fitted(X_batch).reshape(-1)
            if outcome_type == "binary":
                data_loss = F.binary_cross_entropy(
                    prediction.clamp(1e-6, 1.0 - 1e-6), y_batch.reshape(-1)
                )
            else:
                data_loss = F.mse_loss(prediction, y_batch.reshape(-1))
            return data_loss + self.linear_kernel_l2(fitted, self.kernel_l2)

        return self.fit(
            model,
            X,
            treatment,
            y,
            loss_function=supervised_loss,
            stratify=None,
        )


def predict_tensor(
    result: NeuralTrainingResult,
    X: NDArray[np.float64],
) -> object:
    device = next(result.model.parameters()).device
    X_tensor = torch.as_tensor(
        result.scaler.transform(X), dtype=torch.float32, device=device
    )
    result.model.eval()
    with torch.no_grad():
        return result.model(X_tensor)
