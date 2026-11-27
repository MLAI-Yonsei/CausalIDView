# CausalIDView

Experiment code and reproduction assets for **CausalIDView**.

[Paper (v1)](https://arxiv.org/abs/2609.36881v1) · [Code](https://github.com/MLAI-Yonsei/CausalIDView)

[Overview](#overview) · [Installation & usage](#installation--usage) · [Main results](#main-results) · [Code layout](#code-layout) · [Citation](#citation)

## Overview

CausalIDView evaluates conditional average treatment effect (CATE) estimators across identification views, outcome changes, hidden confounding, and randomized trials.

- **Verify the bundled results on CPU.** Start with the [analysis environment](#analysis-environment) and [archive commands](#verify-the-bundled-results-on-cpu).
- **Run the experiments from inputs.** Follow [model setup](#model-setup) and the commands for the experiment you want to rerun. Check [Main results](#main-results) for reference files and how to interpret differences in a fresh run.

## Installation & usage

### Analysis environment

Clone the repository and run the commands from its root. Keep the analysis environment active for orchestration, scoring, and plotting. Model workers use separate environments.

```bash
git clone https://github.com/MLAI-Yonsei/CausalIDView.git
cd CausalIDView
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests
```

### Verify the bundled results on CPU

After installing the analysis environment, run these commands to verify the bundled v2 results and render the figures and table. This workflow needs no GPU, model runtimes, checkpoints, or external datasets. Outputs go to separate archive directories under `artifacts/`. The Figure 6 command scores panels (a) and (b) and the Γ=2 sensitivity results in panel (c).

```bash
python scripts/fig2/verify_fig2.py --archived --output artifacts/fig2_archived/fig2.csv
python scripts/fig2/plot_fig2.py --input artifacts/fig2_archived/fig2.csv --output artifacts/fig2_archived/fig2.pdf
python scripts/fig3/verify_fig3.py --archived
python scripts/fig3/plot_fig3.py --input artifacts/fig3_archived/fig3.csv --output artifacts/fig3_archived/fig3.pdf
python scripts/fig6/verify_fig6.py --archived
python scripts/table2/verify_table2.py --archived --plot
```

Figures 2 and 3 and Table 2 aggregate archived scores. Figure 6 recomputes the scores from archived per-query estimates and ground truth. Figure 6 and Table 2 also accept explicit replacement paths in archive mode.

### Model setup

Use Linux x86-64 with glibc 2.28 or newer and Python 3.10 for the published model profiles. Install Git for the Do-PFN source checkout. For the full run, use a CUDA-capable GPU with compatible drivers.

#### 1. Install the model runtimes

Each model runs in a separate Python environment. Install only the profile you need for a one-world check, or all six for the full figure.

| Runtime | Estimators | Dependencies |
| --- | --- | --- |
| `classical` | KIV, ForestDRIV, XGBoost front-door plug-in, ExtraTrees P-Learner | [classical.txt](requirements/classical.txt) |
| `neural` | Neural-network front-door plug-in and P-Learner | [neural.txt](requirements/neural.txt) |
| `causalpfn` | CausalPFN | [causalpfn.txt](requirements/causalpfn.txt) |
| `dopfn` | Do-PFN | [dopfn.txt](requirements/dopfn.txt) |
| `causalfm` | CausalFM | [causalfm.txt](requirements/causalfm.txt) |
| `tabpfn` | TabPFN-v3.5 estimators | [tabpfn.txt](requirements/tabpfn.txt) |

```bash
for runtime in classical neural causalpfn dopfn causalfm tabpfn; do
    python3.10 -m venv ".venv-${runtime}"
    ".venv-${runtime}/bin/python" -m pip install -r "requirements/${runtime}.txt"
done
```

- **PyTorch wheels.** The archived neural and TabPFN environments use `torch==2.13.0+cu126`. If your package index does not provide it, point `--find-links` at a directory containing that exact wheel. The requirements files pin direct dependencies. Wheel and transitive dependency availability can vary by platform. [Recorded versions](requirements/reference-runtime.json)
- **Existing environments.** Pass `--python-<runtime> /path/to/python` to use an existing interpreter. The launcher checks Python 3.10, relevant package versions, and CUDA availability before fitting. Use `--device cpu` for supported standalone CFM and neural checks. TabPFN-v3.5 uses CUDA.

#### 2. Download and verify the checkpoints

Checkpoints are downloaded separately from the Git clone. The setup scripts fetch them from GitHub or Hugging Face and place them under `Baselines/`. You can run setup on any machine with access to those hosts.

| Model | Upstream weights | Setup |
| --- | --- | --- |
| CausalPFN | [Hugging Face](https://huggingface.co/vdblm/causalpfn) | Figure 2 setup |
| CausalFM (standard, FD, IV binary) | [GitHub](https://github.com/yccm/CausalFM-toolkit) | Figure 2 setup |
| Do-PFN | [GitHub](https://github.com/jr2021/Do-PFN) | Figure 2 setup |
| TabPFN v3.5 | [Hugging Face](https://huggingface.co/Prior-Labs/tabpfn_3_5) | Figure 2 setup, requires login and license acceptance |
| CSA-PFN (MSM) | [Hugging Face](https://huggingface.co/arunkatika/csa-pfn-msm) | [Figure 6(c) setup](#partial-identification-predictions) |

For runs that use TabPFN v3.5, accept the license on its model page, then run `huggingface-cli login` in the analysis environment or provide an authenticated `HF_TOKEN`. Run the following for the full Figure 2 checkpoint set.

```bash
python scripts/fig2/prepare_fig2.py
python scripts/fig2/run_fig2.py --validate-checkpoints
```

Setup verifies all six checkpoint files and both Do-PFN runtime pickles by SHA-256. CausalPFN and CausalFM source snapshots and licenses are included in the repository. [Checkpoint configuration](configs/fig2.yaml)

| Do-PFN asset | Official revision | Verification |
| --- | --- | --- |
| 36 Python source files | [`90d6743`](https://github.com/jr2021/Do-PFN/commit/90d67433b43c4d52d752dc336070f525ff856e0b) | [Git blob hashes](requirements/dopfn-fig2-source.json) |
| `model_submitit_0ccc_id_171b69db_epoch_-1.cpkt`, `dopfn_model.pkl`, `dopfn_config.pkl` | [`9887565`](https://github.com/jr2021/Do-PFN/tree/9887565273fa37e3761abfb2ba2550117eb5320d/artifacts) | [SHA-256 digests](configs/fig2.yaml) |

Do-PFN setup installs the pinned source in `Baselines/CFMs/Do-PFN-fig2/` and can reuse matching weights from `Baselines/CFMs/Do-PFN/artifacts/`.

<details>
<summary>Install one model or use local files</summary>

| Task | Command |
| --- | --- |
| Prepare only CausalPFN | `python scripts/fig2/prepare_fig2.py --only causalpfn` |
| Prepare only Do-PFN | `python scripts/fig2/prepare_fig2.py --only dopfn` |
| Supply a local CausalPFN checkpoint | `python scripts/fig2/prepare_fig2.py --local causalpfn=/path/to/causalpfn_v0.pt` |
| Supply a local copy of the pinned Do-PFN source | `python scripts/fig2/prepare_fig2.py --only dopfn --dopfn-source /path/to/Do-PFN` |

Local files go through the same hash checks.

</details>

### One-world check

This smoke test uses the `classical` runtime and a single SCM world. It runs on CPU without model checkpoints. If you skipped full model setup, install just this runtime first.

```bash
python3.10 -m venv .venv-classical
.venv-classical/bin/python -m pip install -r requirements/classical.txt
```

```bash
python scripts/fig2/generate_data.py
python scripts/fig2/predict_fig2.py --regime IV --model kiv --world 0 --device cpu
python scripts/fig2/run_fig2.py --regime IV --model kiv --world 0 --output artifacts/smoke-kiv.csv
```

For a GPU check with a prepared CausalPFN checkpoint, run `CUDA_VISIBLE_DEVICES=0 python scripts/fig2/predict_fig2.py --regime BD --model causalpfn --world 0`.

<a id="full-gpu-run"></a>

### Figure 2

Use the activated `.venv` environment and complete model setup first. Regenerate the worlds and use a fresh prediction directory when upgrading from a previous version. The figure compares six estimators across four views on 40 paired structural causal model (SCM) worlds. Each world has 1,024 context rows and 100 query rows. Use `PROX` on the command line for the PX panel. [Figure 2 v2 reference](results/fig2.csv)

1. **Generate data and run inference.** The dry run lists the 960 jobs. Set `CUDA_VISIBLE_DEVICES` to select a physical GPU, which the worker sees as `cuda:0`.

   ```bash
   python scripts/fig2/generate_data.py
   python scripts/fig2/verify_fig2_worlds.py
   python scripts/fig2/predict_fig2.py --dry-run
   CUDA_VISIBLE_DEVICES=0 python scripts/fig2/predict_fig2.py
   ```

2. **Score the predictions and render Figure 2.**

   ```bash
   python scripts/fig2/run_fig2.py
   python scripts/fig2/verify_fig2.py
   python scripts/fig2/plot_fig2.py
   ```

The scorer aligns each prediction file with the regenerated SCM world by query ID. It writes world scores and the aggregate table under `artifacts/fig2_fresh/`. The bootstrap intervals measure variability across worlds. The verifier checks the aggregate means, interval endpoints, error components, and best estimator in each view against the bundled v2 reference. It records numerical differences in `fig2_comparison.csv`. The default absolute tolerance is `1e-12`. Use `--atol` to set a different numerical tolerance for your device or runtime.

<details>
<summary>Run subsets or split the work across two GPUs</summary>

Select jobs with `--regime`, `--model`, or repeated `--world` arguments. Disjoint subsets can share a prediction directory. A repeated job replaces that prediction atomically. Give exploratory CPU runs or runs with a different runtime their own `--output` directory.

This example splits the four views between physical GPUs 0 and 1. Start aggregation after both groups finish successfully.

```bash
(CUDA_VISIBLE_DEVICES=0 python scripts/fig2/predict_fig2.py --regime BD &&
 CUDA_VISIBLE_DEVICES=0 python scripts/fig2/predict_fig2.py --regime IV) &
gpu0=$!
(CUDA_VISIBLE_DEVICES=1 python scripts/fig2/predict_fig2.py --regime FD &&
 CUDA_VISIBLE_DEVICES=1 python scripts/fig2/predict_fig2.py --regime PROX) &
gpu1=$!
status0=0
status1=0
wait "$gpu0" || status0=$?
wait "$gpu1" || status1=$?
test "$status0" -eq 0 && test "$status1" -eq 0
```

</details>

### Figure 3

Figure 3 reports spurious response (`E_stay`) and change-tracking error (`E_move`) for the same 24 estimator/view combinations as Figure 2. Each of the 40 worlds has the same 1,024 context units and 100 query units in all three conditions. `stay` reverses the direct confounding contribution to the outcome and preserves CATE. `move` doubles the mediator's outcome coefficient and doubles CATE. The bundled [query IDs and original CATE](results/fig2_world_manifest.json) serve as evaluator targets. The worker fits each condition with the original model and seed, using the observed context outcomes for that condition.

Prepare the [model runtimes and checkpoints](#model-setup), then run:

```bash
python scripts/fig2/generate_data.py
python scripts/fig2/verify_fig2_worlds.py
python scripts/fig3/predict_fig3.py --dry-run
CUDA_VISIBLE_DEVICES=0 python scripts/fig3/predict_fig3.py
python scripts/fig3/run_fig3.py
python scripts/fig3/verify_fig3.py
python scripts/fig3/plot_fig3.py
```

The full run writes 2,880 prediction files under `artifacts/fig3_fresh/predictions/`, with 100 estimates per file. `predict_fig3.py` also accepts `--regime`, `--model`, and repeatable `--world` options for running a subset of jobs. Use a new prediction directory after a change to the code or runtime. The FD Do-PFN route uses a query batch size of 25. The scorer requires complete coverage, checks job identity and the source world fingerprint, and aligns ordered query IDs with the bundled ground truth.

For each world, `E_stay` is the RMS of `stay − original` predictions divided by the RMS of original CATE. `E_move` is the RMS of the difference between the predicted and true `move − original` CATE changes, divided by the same scale. The summary reports means, standard errors, and percentile bootstrap 95% intervals over the 40 worlds. The response plot shows standard errors.

The [Figure 3 reference](results/fig3/reference_manifest.json) uses the 40-world response benchmark with 2,743 retained archived jobs, 17 reruns for missing jobs, and 120 reruns for NN + P-Learner. `verify_fig3.py` checks the two per-world metrics and all summary statistics. It writes `artifacts/fig3_fresh/fig3_comparison.csv` with the actual value, reference value, numerical difference, and verdict for each comparison, then exits nonzero if any value falls outside the default absolute and relative tolerances of `1e-10`. Use `--atol` and `--rtol` to record an explicitly chosen tolerance.

The archive check reproduced all 24 summary rows. A fresh CPU run of KIV across all 40 worlds and three conditions matched both per-world metrics within `8e-13`. Strict agreement of the full fresh GPU run with this reference is unverified.

### Figure 6

This reproduction includes panels (a), (b), and (c). The sensitivity setting for panel (c) is Γ=2.

Generate the predictions for the panel you want to check. Each verifier uses that panel's held-out targets and bundled v2 reference. CATE containment uses the true CATE.

#### Synthetic point predictions

Figure 6(a) uses the Figure 2 worlds, runtimes, and checkpoints. The observed input includes all 51 covariates and the hidden-confounding input omits the final confounder.

```bash
python scripts/fig2/generate_data.py
CUDA_VISIBLE_DEVICES=0 python scripts/fig6/predict_fig6a.py
```

The runner writes `artifacts/fig6a_predictions/`. Use `--dry-run` to list the 480 jobs. For a small run, use `--world 0 --model tabpfn_x` and a separate `--predictions` directory.

#### IHDP-HC point predictions

Figure 6(b) uses the three CFM runtimes from [Model setup](#model-setup) and a separate TabPFN 9.2.0 environment. This environment shares the pinned v3.5 checkpoint with Figure 2.

```bash
python3.10 -m venv .venv-fig6b
.venv-fig6b/bin/python -m pip install -r requirements/tabpfn-fig6b.txt
.venv-fig6b/bin/python scripts/fig2/prepare_fig2.py --only tabpfn_v35

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
  .venv-fig6b/bin/python scripts/fig6/predict_fig6b.py
CUDA_VISIBLE_DEVICES=0 python scripts/fig6/predict_fig6b_cfms.py
```

Both runners write model-specific files under `artifacts/fig6b_predictions/`. Each of the 100 trials uses 470 training rows followed by 202 validation rows and 75 query rows. The Hidden view omits maternal marital status. The scorer aligns the files with the evaluator-only sample IDs. For a smoke check, use `--trials 1` and a separate `--predictions` directory. For parallel inference, start one process per physical GPU with a common `--shards` count and distinct zero-based `--shard` numbers.

The CFM smoke run used trial 0 in both conditions. The largest absolute sPEHE differences from the archive were `4.7e-6` for CausalPFN, `1.9e-7` for CausalFM, and `0.00156` for Do-PFN. TabPFN 9.2.0 rerun scores can also differ from the archive.

#### Partial-identification predictions

The bound runner reads the 40 frozen observed worlds under `data/fig6/bound_worlds/` and uses the estimators in `Baselines/src/baselines/partial_identification/`. The worlds are ready to use.

```bash
.venv-classical/bin/python scripts/fig6/predict_fig6_bounds.py --model logistic
.venv-classical/bin/python scripts/fig6/predict_fig6_bounds.py --model xgboost
```

Set up the remaining four methods with the TabPFN 9.0.0 runtime. The setup commands install the shared v3.5 checkpoint, the public CSA-PFN checkpoint, and a pinned NeuralCSA checkout with the experiment's device compatibility patch. They place these assets under `Baselines/`. The B-Learner source and license are included.

```bash
python3.10 -m venv .venv-fig6-bounds
.venv-fig6-bounds/bin/python -m pip install -r requirements/fig6-bounds.txt
.venv-fig6-bounds/bin/python scripts/fig2/prepare_fig2.py --only tabpfn_v35
.venv-fig6-bounds/bin/python scripts/fig6/prepare_fig6_bounds.py

for model in tabpfn_v35 b_learner_tabpfn_all csa_pfn_msm neuralcsa; do
  CUDA_VISIBLE_DEVICES=0 .venv-fig6-bounds/bin/python scripts/fig6/predict_fig6_bounds.py \
    --model "$model" --device cuda:0
done
```

B-Learner and NeuralCSA fit on each world's context rows before predicting its query bounds. Omitting `--world` runs all 40 worlds. Each model writes a new CSV under `artifacts/fig6_bounds/`. The sensitivity CSVs contain Γ=2 endpoints. For a one-world check, add `--world 0 --output /tmp/fig6-bounds.csv`.

#### Score and compare each panel

```bash
python scripts/fig6/verify_fig6a.py
python scripts/fig6/verify_fig6b.py
python scripts/fig6/verify_fig6c.py
```

| Panel | Default prediction inputs | Output | v2 comparison |
| --- | --- | --- | --- |
| (a) | `artifacts/fig6a_predictions/` | `artifacts/fig6a_fresh/` | [Mean sPEHE and bootstrap interval](results/fig6/point_summary.csv) |
| (b) | `artifacts/fig6b_predictions/` | `artifacts/fig6b_fresh/` | [Mean sPEHE and bootstrap interval](results/fig6/point_summary.csv) |
| (c) | `artifacts/fig6_bounds/` | `artifacts/fig6c_fresh/` | [Mean endpoint RMSE and standard deviation](results/fig6/panel_c.csv) |

Each command writes numerical CSVs and a source manifest. Use `--predictions` for a different input directory for (a) or (b), or repeat it for individual prediction files for (c). Use `--cfm-predictions` for (b) when the CFM files are in a separate directory. Add `--atol` with your absolute tolerance to require agreement for the values in `comparison.csv`. Use `--archived` to check a panel from saved predictions. [Input and reference provenance](data/fig6/PROVENANCE.md)

Run `python scripts/fig6/verify_fig6.py` to evaluate all three panels together and render the figure. Add `--paper-reference` for the historical comparison against the PDF bar heights.

### Table 2

The Table 2 predictor runs all ten estimators. Causal Tree, TARNet, GRF, and CFRNet are fitted from scratch on each primary split. All ten are scored against a null CATE of zero.

Download the three raw datasets and put the files directly under the directory passed to `--raw-dir`. The preparation command checks their SHA-256 hashes.

| File | Download | File to extract |
| --- | --- | --- |
| `ACTG175.txt` | [ACTG175 text file](https://raw.githubusercontent.com/mjuraska/speff2trial/master/data/ACTG175.txt) | Save as `ACTG175.txt` |
| `Recsfile.dta` | [Pennsylvania ZIP](https://www.upjohn.org/sites/default/files/2019-02/PA_ReempBonus.zip) | `cd/data/Upjohn Institute Created files/Recsfile.dta` |
| `illinois.dta` | [Illinois ZIP](https://www.upjohn.org/sites/default/files/2019-04/il_ui_experiment_data-images-install.zip) | `data/illinois.dta` |

```text
/path/to/raw/
├── ACTG175.txt
├── Recsfile.dta
└── illinois.dta
```

Use the CFM and TabPFN runtimes from model setup and the `.venv-neural` runtime for TARNet and CFRNet. [Table 2 input provenance and checksums](data/table2/provenance.json)

Causal Tree and GRF run through `Rscript`. The local verification used R 4.4.3. Install the pinned R packages in the repository's `.r_libs/` or supply an existing installation with `--rscript` and `--r-libs`. The R adapters verify the upstream commits at runtime. [R source pins](Baselines/src/baselines/r_backend/sources.py)

```bash
mkdir -p .r_libs
export R_LIBS_USER="$PWD/.r_libs"
Rscript --vanilla - <<'RS'
lib <- Sys.getenv("R_LIBS_USER")
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages(c("jsonlite", "remotes"), lib = lib)
remotes::install_github("susanathey/causalTree",
  ref = "51a8762bff0ab17b971bce852cb37bece2d8240f", lib = lib, upgrade = "never")
remotes::install_github("grf-labs/grf",
  ref = "5bee99b51471f76cb2d63acbc8a9b0ffec408ba0",
  subdir = "r-package/grf", lib = lib, upgrade = "never")
RS
```

```bash
python -m pip install -r requirements/table2.txt
python scripts/table2/predict_table2.py --raw-dir /path/to/raw --prepare-only
CUDA_VISIBLE_DEVICES=0 python scripts/table2/predict_table2.py --raw-dir /path/to/raw
```

The predictor verifies all 120 input and split hashes and writes `rct_null_results.csv` and `requested_baseline_results.csv` under `artifacts/table2/fresh/`. Use `--dataset actg175 --realization 0` and a separate `--output-dir` to check one full-size split. Repeat `--model` to choose specific estimators. Causal Tree and GRF run on CPU. To train TARNet and CFRNet on CPU, select `--neural-device cpu`.

The traditional training code follows the CausalArena Table 2 implementation. TARNet and CFRNet average predictions from three optimization seeds. The seed formula and model order are defined in the adapter, and the neural architecture uses the bundled default profile. [Training adapter](Baselines/src/baselines/table2.py) · [Default profile](configs/baselines/default.yaml) · [Original run records](data/table2/requested_baseline_results.csv)

```bash
python scripts/table2/verify_table2.py --plot
```

The verifier checks both score files and the pinned split hashes. It squares each realization's metrics before computing the mean and population standard deviation. It saves the table and `table2_comparison.csv` under `artifacts/table2/fresh/`. The `matches` column checks agreement with the bundled reference, including the displayed values and full-precision means and standard deviations at the default absolute tolerance of `1e-12`. The `display_matches` column reports agreement at the four-decimal precision shown in the table. [Table 2 v2 reference](results/table2/table2.csv)

Use `--archived` to rebuild the full table from the bundled scores. Local CFRNet reruns can differ from the archive. On Illinois realization 0, the original runner and the bundled implementation produced identical predictions locally, with null sPEHE of `0.12818` compared with `0.10509` in the archive. The comparison CSV reports the measured changes for a fresh run.

## Main results

Compare fresh outputs against the bundled v2 references. A verifier can write a comparison CSV and then exit with an error because the results differ. If the report exists, use it to distinguish a numerical mismatch from missing inputs or an incomplete run.

| Experiment | Bundled v2 reference | Fresh comparison under `artifacts/` | How the fresh verifier handles differences |
| --- | --- | --- | --- |
| [Figure 2](#figure-2) | [fig2.csv](results/fig2.csv) | `fig2_fresh/fig2_comparison.csv` | Fails above the default absolute tolerance of `1e-12`. Strict agreement for the full corrected GPU run remains open. |
| [Figure 3](#figure-3) | [Figure 3 CSVs](results/fig3/) | `fig3_fresh/fig3_comparison.csv` | Fails when comparisons exceed the default absolute and relative tolerances of `1e-10`. The full fresh GPU run remains unverified against this reference. |
| [Figure 6](#figure-6) | [Figure 6 CSVs](results/fig6/) | `fig6a_fresh/comparison.csv`, `fig6b_fresh/comparison.csv`, `fig6c_fresh/comparison.csv` | Panel verifiers report numerical differences by default. Pass `--atol` to require agreement. |
| [Table 2](#table-2) | [table2.csv](results/table2/table2.csv) | `table2/fresh/table2_comparison.csv` | Fails on mismatches and reports both full-precision and four-decimal comparisons. Fresh CFRNet scores can differ from the archive. |

<details>
<summary><strong>Figure 2 verification details</strong></summary>

| Check | Result |
| --- | --- |
| Aggregation of archived metrics | All 24 rows match the bundled Figure 2 reference exactly. |
| Regenerated SCM worlds | All 40 match the archived query covariate hashes, IDs, and true CATE values. |
| Pinned Do-PFN on RTX 4090 | Across 160 jobs, the maximum aggregate delta was `3.09e-5` over the mean, confidence interval endpoints, and error components. |
| Model order using combined results | All four views match the bundled Figure 2 order when the pinned Do-PFN scores replace the Do-PFN scores in the saved 960-job run that predates the SCM arithmetic correction. |
| Full GPU rerun with the corrected SCM and pinned Do-PFN | Strict numerical verification at `1e-12` remains open. |

</details>

## Code layout

| Path | What you will find |
| --- | --- |
| [`data/`](data/) | SCM and world generation, frozen IHDP-HC inputs, Figure 6 predictions and targets, and Table 2 replicate results |
| [`Baselines/src/baselines/`](Baselines/src/baselines/) | Model adapters, the inference worker, and estimator implementations |
| [`Baselines/CFMs/`](Baselines/CFMs/) | Causal model source snapshots and checkpoint layout |
| [`configs/`](configs/) and [`requirements/`](requirements/) | Experiment settings, runtime profiles, and the Do-PFN source manifest |
| [`scripts/fig2/`](scripts/fig2/) | Figure 2, shared SCM world generation, and checkpoint setup |
| [`scripts/fig3/`](scripts/fig3/) | Figure 3 prediction, scoring, verification, and plots |
| [`scripts/fig6/`](scripts/fig6/) | Figure 6(a), (b), and (c) prediction and verification |
| [`scripts/table2/`](scripts/table2/) | Table 2 input preparation, prediction, and verification |
| [`results/`](results/) | Bundled v2 reference tables and Figure 2 world manifest |
| `artifacts/` | Generated worlds, predictions, and reports from your run |

<details>
<summary>Data boundaries and source provenance</summary>

- **Predictions.** Files contain ordered query `unit_id` and `tau_hat` arrays. The evaluator checks order, shape, and finiteness. The worker receives observed context and configured query features. Oracle outcomes and CATE are reserved for evaluation.
- **Model inputs.** FD CausalPFN and Do-PFN use X alone as covariates in both splits. Native CausalFM receives the auxiliary variable in context and X in the query. PX modular estimators receive proxies in context and X in the query.
- **Estimator code.** The worker runs as `baselines.fig2_worker`. The front-door plug-in lives in `frontdoor.py` and shares `neural.py` with the proximal estimators. The neural front-door route uses the production OOF integration and MLP factories. TabPFN fitting routines and seeds live in `tabpfn/fig2.py`.
- **References.** CausalIDView commit `b884c4951f6acd00f129064038aa82adbb2754b6` provides the published CSV and PDF. Estimator code and world metrics come from CausalIDView_Rebuttal `6d0387f78`. The released world metrics omit machine-specific paths, and the 40 query manifests come from the BD CausalPFN records. The transfer provides per-world metrics and query manifests. The original per-query predictions are unavailable.
- **Figure 2 and Appendix Table 7.** The means for 23 cells agree to three decimals. For FD TabPFN-v3.5, Figure 2 records `0.15339920346215882` and Table 7 Default displays `0.154`. The verifier uses the bundled Figure 2 CSV.

</details>

## Citation

Cite the [CausalIDView paper (arXiv:2609.36881v1)](https://arxiv.org/abs/2609.36881v1) when using this code or the benchmark results.
