"""Deterministic SCM used by the Figure 2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np
from scipy.special import expit


CATE_FAMILIES = (
    "linear", "quadratic_interaction", "threshold_piecewise",
    "fourier_nonstationary", "shallow_mlp",
)


@dataclass(frozen=True)
class World:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def namespace_seed(world_seed: int, namespace: str) -> int:
    """Historical base-world namespace retained for bitwise paper compatibility."""
    payload = f"master-scm-v1::{world_seed}::{namespace}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 2**32


def _rng(seed: int, namespace: str) -> np.random.Generator:
    return np.random.default_rng(namespace_seed(seed, namespace))


def _modulation_rng(seed: int) -> np.random.Generator:
    payload = f"master-scm-v2::{seed}::outcome_confounding_modulation".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 2**32
    return np.random.default_rng(value)


def _unit_vector(rng: np.random.Generator, d: int, active: int) -> np.ndarray:
    indices = rng.choice(d, size=min(active, d), replace=False)
    weights = np.zeros(d, dtype=np.float64)
    weights[indices] = rng.normal(size=len(indices))
    return weights / np.linalg.norm(weights)


def _raw_cate(x: np.ndarray, family: str, p: Mapping[str, np.ndarray]) -> np.ndarray:
    a, b = x @ p["a"], x @ p["b"]
    if family == "linear":
        return a
    if family == "quadratic_interaction":
        return 0.7 * a + 0.45 * a * b + 0.25 * (b * b - 1.0)
    if family == "threshold_piecewise":
        return 0.75 * (a > 0) - 0.55 * (b > 0.4) + 0.25 * a
    if family == "fourier_nonstationary":
        return np.sin(1.4 * a) + 0.45 * np.cos(2.1 * b) + 0.18 * a * b
    if family == "shallow_mlp":
        return np.tanh(x @ p["hidden_w"].T + p["hidden_b"]) @ p["out_w"]
    raise ValueError(f"unknown CATE family: {family}")


def _cate_parameters(seed: int, d: int, family: str) -> dict[str, np.ndarray]:
    rng = _rng(seed, "cate_parameters")
    result = {"a": _unit_vector(rng, d, 10), "b": _unit_vector(rng, d, 10)}
    if family == "shallow_mlp":
        result["hidden_w"] = np.stack([_unit_vector(rng, d, 12) for _ in range(8)])
        result["hidden_b"] = rng.normal(scale=0.35, size=8)
        result["out_w"] = rng.normal(size=8) / math.sqrt(8)
    return result


def _calibrate_intercept(logits: np.ndarray) -> float:
    low, high = -8.0, 8.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if float(np.mean(expit(logits + middle))) < 0.5:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def sample_world(world_seed: int, config: Mapping[str, Any]) -> World:
    """Generate one paired world without importing any parent repository."""
    scm = config["scm"]
    d = int(scm["d_x"])
    n_context, n_query = int(scm["n_context"]), int(scm["n_query"])
    n, n_cal = n_context + n_query, int(scm["calibration_population"])
    family = CATE_FAMILIES[world_seed % len(CATE_FAMILIES)]

    x = _rng(world_seed, "X").normal(size=(n, d))
    x_cal = _rng(world_seed, "X_calibration").normal(size=(n_cal, d))
    cate_p = _cate_parameters(world_seed, d, family)
    raw_cal = _raw_cate(x_cal, family, cate_p)
    tau = float(scm["cate_mean"]) + float(scm["cate_sd"]) * (
        (_raw_cate(x, family, cate_p) - raw_cal.mean()) / raw_cal.std(ddof=0)
    )

    rng = _rng(world_seed, "nuisance_parameters")
    mu_w = _unit_vector(rng, d, 14)
    treatment_w = _unit_vector(rng, d, 14)
    proxy_z_w = _unit_vector(rng, d, 8)
    proxy_w_w = _unit_vector(rng, d, 8)
    mu = 0.7 * (x @ mu_w) + 0.25 * np.sin(x @ np.roll(mu_w, 1))
    u = np.where(_rng(world_seed, "U").random(n) < 0.5, -1, 1).astype(np.int8)
    instrument = (_rng(world_seed, "I").random(n) < 0.5).astype(np.int8)

    kappa = float(scm["hidden_severity_kappa"])
    alpha_u, alpha_i = kappa * float(scm["alpha_u"]), float(scm["alpha_i"])
    treatment_scale = float(scm["treatment_logit_scale"])
    cal_x = x_cal @ treatment_w
    cal_logits = np.concatenate([
        treatment_scale * cal_x + alpha_u * latent + alpha_i * inst
        for latent in (-1, 1) for inst in (0, 1)
    ])
    intercept = _calibrate_intercept(cal_logits)
    base = intercept + treatment_scale * (x @ treatment_w) + alpha_u * u
    p0, p1 = expit(base), expit(base + alpha_i)
    uniform = _rng(world_seed, "V_T").random(n)
    t0, t1 = (uniform <= p0).astype(np.int8), (uniform <= p1).astype(np.int8)
    treatment = np.where(instrument == 1, t1, t0).astype(np.int8)

    proxy_scale = float(scm["proxy_x_scale"])
    pz = expit(proxy_scale * (x @ proxy_z_w) + float(scm["proxy_lambda_z"]) * u)
    pw = expit(proxy_scale * (x @ proxy_w_w) + float(scm["proxy_lambda_w"]) * u)
    z_proxy = (_rng(world_seed, "proxy_Z_noise").random(n) <= pz).astype(np.int8)
    w_proxy = (_rng(world_seed, "proxy_W_noise").random(n) <= pw).astype(np.int8)

    delta = float(scm["mediator_delta"])
    m_noise = float(scm["mediator_noise_sd"]) * _rng(world_seed, "epsilon_M").normal(size=n)
    y_noise = float(scm["outcome_noise_sd"]) * _rng(world_seed, "epsilon_Y").normal(size=n)
    m0, m1 = m_noise, delta + m_noise
    beta = tau / delta
    gamma_base = kappa * float(scm["gamma_u"])
    # Preserve the historical two-stage arithmetic exactly: the canonical
    # release formed a base outcome and recovered its realized noise before
    # applying the X-dependent U -> Y modulation.
    base_y0 = mu + gamma_base * u + beta * m0 + y_noise
    recovered_y_noise = base_y0 - mu - gamma_base * u - beta * m0
    mod = config["outcome_confounding_modulation"]
    direction_rng = _modulation_rng(world_seed)
    direction = _unit_vector(direction_rng, d, int(mod["active_features"]))
    gamma_x = gamma_base * (1.0 + float(mod["x_scale"]) * np.tanh(x @ direction))
    y0 = mu + gamma_x * u + beta * m0 + recovered_y_noise
    y1 = mu + gamma_x * u + beta * m1 + recovered_y_noise
    mediator = np.where(treatment == 1, m1, m0)
    outcome = np.where(treatment == 1, y1, y0)

    permutation = _rng(world_seed, "split_permutation").permutation(n)
    split = np.ones(n, dtype=np.int8)
    split[permutation[:n_context]] = 0
    arrays = {
        "world_id": np.full(n, world_seed, dtype=np.int32),
        "unit_id": world_seed * 1_000_000 + np.arange(n, dtype=np.int64),
        "X": x, "U": u, "I": instrument, "Z_P": z_proxy, "W_P": w_proxy,
        "T": treatment, "M": mediator, "Y": outcome,
        "T0": t0, "T1": t1, "M0": m0, "M1": m1,
        "Y0": y0, "Y1": y1, "tau": tau, "beta": beta,
        "mu": mu, "gamma_u_x": gamma_x, "split": split,
    }
    metadata = {
        "protocol": config["protocol"], "world_seed": world_seed,
        "cate_family": family, "n_context": n_context, "n_query": n_query, "d_x": d,
    }
    return World(arrays=arrays, metadata=metadata)
