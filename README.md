# CausalIDView

This compact repository reproduces two CausalIDView paper figures from the
bundled, frozen evaluation outputs:

- `figure2_point_estimation_benchmark`
- `fig_fd_structural_response_combined`

No external dataset, model checkpoint, GPU, or absolute local path is needed.
The renderer requires Python 3.10 or newer; exact Python package versions are
pinned in `requirements.txt`.

## Run

```bash
git clone https://github.com/goyaground/CausalIDView.git
cd CausalIDView

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

Outputs are written to `figures/` in PDF, PNG, and SVG format. The exact
aggregate values used for plotting are also written as CSV files.

Run only one figure if needed:

```bash
python reproduce.py --figure figure2
python reproduce.py --figure fd-response
```

Verify the bundled source checksums and numerical contracts without rendering:

```bash
python reproduce.py --verify-only
```

Use a different output directory:

```bash
python reproduce.py --output /path/to/output
```

## Repository contents

```text
CausalIDView/
├── README.md
├── requirements.txt
├── reproduce.py
└── data/
    ├── manifest.json
    ├── figure2_world_metrics.csv
    └── fd_structural_response.npz
```

- `figure2_world_metrics.csv` contains the 40-world metrics for the 24 selected
  model–regime configurations. Figure 2 statistics and confidence intervals are
  recomputed from these world-level records.
- `fd_structural_response.npz` contains the aligned query CATE, original
  predictions, and causal-response predictions required for the FD response
  figure (40 worlds × 100 queries).
- `manifest.json` records source provenance, SHA-256 checksums, dimensions, and
  fixed numerical checks.

## Reproducibility scope

This release reproduces the reported numerical summaries and figures from the
immutable evaluated predictions. It does not retrain the foundation models or
regenerate their predictions. The compact evaluated data are included so that
figure reproduction is CPU-only and independent of private filesystem layouts.

The project was previously developed under the names `CausalArena` and
`MasterSCM`; those names remain only in the provenance recorded in
`data/manifest.json`.
