# CausalIDView: Figure 2 reproduction

This anonymous artifact is scoped to the point-estimation benchmark in Figure 2.
It has no runtime or filesystem dependency on a parent project, and contains no
absolute machine path. Python 3.10 or newer is required.

## Quick reproduction (CPU only)

The committed `results/fig2.csv` is the frozen 24-cell plotting table. Rebuild
the paper PDF directly:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/plot_fig2.py
```

This writes `figures/fig2.pdf`. To produce a high-resolution comparison image as
well, add `--png figures/fig2.png`.

## Full evaluation path

The full data-to-metric path is separated from the quick renderer:

```text
SCM generation -> observational views -> estimator inference
               -> aligned query predictions -> metrics
               -> results/fig2.csv -> figures/fig2.pdf
```

Generate all 40 deterministic worlds:

```bash
python scripts/generate_data.py
```

Estimator implementations must consume only the regime-specific contracts in
`data/estimators/`. They write one file per model and world:

```text
artifacts/predictions/<REGIME>/<MODEL>/world_000.npz
```

Each file contains exactly two arrays: `unit_id` and `tau_hat`. The evaluator
rejects a file if its query IDs are not in the SCM query order. After inference:

```bash
python scripts/run_fig2.py
python scripts/plot_fig2.py
```

`run_fig2.py` recomputes per-world sPEHE², its ATE-error² and centered-sPEHE²
decomposition, and the fixed 10,000-replicate world bootstrap. Ground truth is
kept in the evaluator and is never included in an estimator view.

The foundation-model source trees and weights are not vendored: their upstream
licenses and environments differ, and TabPFN-v3 access is gated. The exact
revision, filename, local relative path, and expected SHA-256 for every weight
are in `configs/fig2.yaml`. Validate downloaded files with:

```bash
python scripts/run_fig2.py --validate-checkpoints
```

The command continues to prediction evaluation after validation, so missing
prediction artifacts will then be reported explicitly. It never substitutes a
surrogate model or the frozen plotting table for missing inference.

## Original-author model and checkpoint sources

| Model | Official source | Checkpoint source used by this benchmark |
|---|---|---|
| CausalPFN | [vdblm/CausalPFN](https://github.com/vdblm/CausalPFN) | [`causalpfn_v0.pt`](https://huggingface.co/vdblm/causalpfn/blob/ccfc5083f28270d09356b8c35190073df17798d5/causalpfn_v0.pt) |
| Do-PFN | [jr2021/Do-PFN](https://github.com/jr2021/Do-PFN) | [bundled author checkpoint](https://github.com/jr2021/Do-PFN/blob/9887565273fa37e3761abfb2ba2550117eb5320d/artifacts/model_submitit_0ccc_id_171b69db_epoch_-1.cpkt) |
| CausalFM | [yccm/CausalFM](https://github.com/yccm/CausalFM) | [official toolkit checkpoints](https://github.com/yccm/CausalFM-toolkit/tree/bb74ef729c70d274fe2bb422c3b0edff4754fa37) |
| TabPFN-v3 | [PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN) | [official gated Hugging Face repository](https://huggingface.co/Prior-Labs/tabpfn_3/tree/24a16a89d245878b846555110985634aa2e656d7) |

For Do-PFN, the upstream artifact name ends in `epoch_-1`; the upstream metadata
does not establish that it is a validation-selected “best” checkpoint, so this
repository makes no such claim. TabPFN-v3 requires accepting its official
license and authenticating with Hugging Face before download.

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

Run the contract and leakage tests with `pytest -q`.
