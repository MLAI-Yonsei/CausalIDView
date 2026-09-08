# CausalIDView

This compact repository reproduces three CausalIDView paper figures from the
bundled, frozen evaluation outputs:

- `figure2_point_estimation_benchmark`
- `hidden_confounder_track`
- `fig_fd_structural_response_combined`

No external dataset, model checkpoint, GPU, or absolute local path is needed.
The renderer requires Python 3.10 or newer; exact Python package versions are
pinned in `requirements.txt`.

## Run

```bash
cd CausalIDView

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

Outputs are written to `figures/`. Every figure is emitted as PDF and PNG;
Figure 2 and the FD response also include SVG, while the hidden-confounder
track includes JPEG aliases. The exact values used for plotting are written as
CSV files.

Run only one figure if needed:

```bash
python reproduce.py --figure figure2
python reproduce.py --figure hidden-confounder
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
├── hidden_confounder.py
├── figures/
│   ├── hidden_confounder_track.pdf
│   ├── hidden_confounder_track.png
│   └── hidden_confounder_track.jpg
└── data/
    ├── manifest.json
    ├── figure2_world_metrics.csv
    ├── fd_structural_response.npz
    ├── hidden_confounder_track_manifest.json
    └── hidden_confounder_track_plot_data.csv
```

- `figure2_world_metrics.csv` contains the 40-world metrics for the 24 selected
  model–regime configurations. Figure 2 statistics and confidence intervals are
  recomputed from these world-level records.
- `fd_structural_response.npz` contains the aligned query CATE, original
  predictions, and causal-response predictions required for the FD response
  figure (40 worlds × 100 queries).
- `hidden_confounder_track_plot_data.csv` contains the 376 frozen independent
  observations and aggregate rows used by the four hidden-confounder panels.
  Its companion manifest records the panel source families and replication
  units.
- `manifest.json` records source provenance, SHA-256 checksums, dimensions, and
  fixed numerical checks.

## Reproducibility scope

This release reproduces the reported numerical summaries and figures from the
immutable evaluated predictions. It does not retrain the foundation models or
regenerate their predictions. The compact evaluated data are included so that
figure reproduction is CPU-only and independent of private filesystem layouts.
