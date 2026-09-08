# CausalIDView

CausalIDView is a reproducibility scaffold for evaluating causal effect
estimators under different identification settings while holding the underlying
structural causal model (SCM) fixed. It separates data generation, observable
information, estimator inference, ground-truth evaluation, and visualization so
that each stage can be inspected independently. The current release includes the
frozen results and scripts used to reproduce the paper's Figure 2 benchmark.

This anonymous artifact is self-contained: it has no runtime or filesystem
dependency on a parent project and contains no absolute machine path. Python
3.10 or newer is required.

## Overview

One deterministic SCM generates paired worlds shared across four identification
regimes:

- **Back-door (BD):** observed covariates include the adjustment variable.
- **Front-door (FD):** the mediator is observed.
- **Instrumental variable (IV):** the instrument is observed.
- **Proximal (PROX):** treatment- and outcome-inducing proxies are observed.

All regimes share the same covariates, treatment assignment, potential outcomes,
unit IDs, context/query split, and conditional treatment-effect target. Only the
information exposed to an estimator changes. Query potential outcomes and true
treatment effects remain evaluator-only.

The repository follows two complementary workflows:

```text
Full evaluation
SCM generation -> observational views -> estimator inference
               -> aligned predictions -> metrics -> aggregate results -> plot

Quick reproduction
frozen aggregate results -> plot
```

## Quick reproduction

The committed aggregate table allows the benchmark plot to be rebuilt without
model checkpoints or a GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/plot_fig2.py
```

The output is written to `figures/fig2.pdf`. A high-resolution image can also be
created with:

```bash
python scripts/plot_fig2.py --png figures/fig2.png
```

## Full evaluation

Generate the 40 deterministic paired worlds:

```bash
python scripts/generate_data.py
```

Estimator implementations consume only the regime-specific contracts in
`data/estimators/` and write one output per model and world:

```text
artifacts/predictions/<REGIME>/<MODEL>/world_000.npz
```

Each prediction file must contain `unit_id` and `tau_hat`. The evaluator rejects
misaligned query IDs rather than reordering or silently matching predictions.
After estimator inference, run:

```bash
python scripts/run_fig2.py
python scripts/plot_fig2.py
```

The evaluation computes per-world sPEHE² and its exact decomposition into
ATE-error² and centered-sPEHE². Uncertainty is calculated with a fixed
10,000-replicate bootstrap whose sampling unit is the SCM world.

## Foundation-model provenance

Foundation-model source trees and weights are not redistributed. Their upstream
licenses and execution environments differ, and TabPFN-v3 access is gated. The
exact revision, filename, repository-relative destination, and expected SHA-256
for every checkpoint are recorded in `configs/fig2.yaml`.

| Model | Official implementation | Checkpoint source |
|---|---|---|
| CausalPFN | [vdblm/CausalPFN](https://github.com/vdblm/CausalPFN) | [`causalpfn_v0.pt`](https://huggingface.co/vdblm/causalpfn/blob/ccfc5083f28270d09356b8c35190073df17798d5/causalpfn_v0.pt) |
| Do-PFN | [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN) | [author-provided checkpoint](https://github.com/jr2021/Do-PFN/blob/9887565273fa37e3761abfb2ba2550117eb5320d/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt) |
| CausalFM | [yccm/CausalFM](https://github.com/yccm/CausalFM) | [official toolkit checkpoints](https://github.com/yccm/CausalFM-toolkit/tree/bb74ef729c70d274fe2bb422c3b0edff4754fa37) |
| TabPFN-v3 | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) | [official gated model repository](https://huggingface.co/Prior-Labs/tabpfn_3/tree/24a16a89d245878b846555110985634aa2e656d7) |

For Do-PFN, the upstream artifact is named with `epoch_-1`; the available
upstream metadata does not establish that it is a validation-selected “best”
checkpoint, so no such claim is made here. TabPFN-v3 requires acceptance of its
official license and Hugging Face authentication.

Downloaded checkpoints can be validated before evaluation:

```bash
python scripts/run_fig2.py --validate-checkpoints
```

Missing or checksum-mismatched checkpoints and prediction artifacts produce
explicit errors. The evaluation never substitutes a surrogate estimator or the
frozen aggregate table for missing model inference.

## Metrics and safeguards

- The unit of replication and bootstrap uncertainty is the independently drawn
  SCM world.
- Potential outcomes and true CATE values are isolated from estimator inputs.
- Context rows expose observed treatment and outcome; query rows expose only
  variables permitted by the selected identification regime.
- Prediction files are joined by exact query `unit_id` order.
- The numerical identity `sPEHE² = ATE-error² + centered-sPEHE²` is tested.
- Generated worlds, downloaded checkpoints, and intermediate predictions are
  excluded from version control.

Run the identification, leakage, metric, and frozen-table contract tests with:

```bash
pytest -q
```

## Repository layout

```text
CausalIDView/
├── README.md
├── requirements.txt
├── data/
│   ├── __init__.py
│   ├── scm.py
│   ├── generate_world.py
│   ├── views.py
│   ├── ground_truth.py
│   ├── metrics.py
│   └── estimators/
│       ├── __init__.py
│       ├── backdoor.py
│       ├── frontdoor.py
│       ├── iv.py
│       └── proximal.py
├── configs/
│   └── fig2.yaml
├── scripts/
│   ├── generate_data.py
│   ├── run_fig2.py
│   └── plot_fig2.py
├── results/
│   └── fig2.csv
├── figures/
│   └── fig2.pdf
└── tests/
    └── test_identification.py
```
