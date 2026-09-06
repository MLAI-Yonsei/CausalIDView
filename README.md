# CausalIDView

CausalIDView benchmarks CATE estimators under multiple observational
identification regimes generated from a shared structural causal model:
back-door, front-door, instrumental variable, proximal/proxy control, and
hidden confounding.

> The project was previously named CausalArena. Some code paths and artifact
> metadata still use `CausalArena` or `MasterSCM` for compatibility.

## Setup

- Requirements: Linux, Python 3.10+, and an NVIDIA GPU for GPU-based models.
- Tested GPU environment: CUDA 12.6 and NVIDIA RTX A6000.
- Install the pinned dependencies:

  ```bash
  git clone https://github.com/goyaground/CausalIDView.git
  cd CausalIDView

  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r Baselines/requirements.lock
  python -m pip install --no-deps -e Baselines

  export PYTHONPATH="$PWD/MasterSCM:$PWD/Baselines/src:$PWD/Baselines/TabPFN/cate_estimators/src"
  export TABPFN_DISABLE_TELEMETRY=1
  ```

- Exact model-specific environments and checkpoint paths are listed in
  `MasterSCM/configs/main.yaml` and `MasterSCM/models/`.

## Data

- The main benchmark is fully synthetic; no external raw data are required.
- Structural equations and settings are in `MasterSCM/master_scm/` and
  `MasterSCM/configs/main.yaml`.
- Generate the 40 benchmark worlds:

  ```bash
  python -m master_scm.cli generate --config MasterSCM/configs/main.yaml
  ```

- Generated worlds are saved in `MasterSCM/data/master_worlds/world_000/`
  through `world_039/`. Each world contains:
  - `full_world.parquet`: realized full-data SCM
  - `potentials.npz`: potential outcomes and oracle targets
  - `structural_metadata.json`: structural parameters
  - `splits.json`: context/query split
  - `view_hashes.json`: alignment hashes
- Model-ready arrays are stored in `MasterSCM/data/model_inputs/`.
- Predictions and metadata are stored in `MasterSCM/results/`,
  `MasterSCM/results_candidate/`, and `MasterSCM/results_corrected/`.

### IHDP/IHDP-HC

- The external data file is `MasterSCM/data/ihdp_quince/ihdp.RData`.
- If it is not included in the release:

  ```bash
  mkdir -p MasterSCM/data/ihdp_quince
  curl -L \
    https://github.com/vdorie/npci/raw/master/examples/ihdp_sim/data/ihdp.RData \
    -o MasterSCM/data/ihdp_quince/ihdp.RData
  ```

## Reproduce the paper figures

```bash
python MasterSCM/main_fig/reproduce.py
```

- This command regenerates the current figures and numerical tables from saved
  prediction artifacts.
- Outputs are written to `figures/main_fig/`.
- Read-only verification:

  ```bash
  python MasterSCM/main_fig/reproduce.py --verify-only
  python MasterSCM/main_fig/reproduce.py --verify-only --strict-render-hash
  ```

## Reproduce model results

- CausalPFN, Do-PFN, and CausalFM:

  ```bash
  python -m master_scm.runner smoke --config MasterSCM/configs/main.yaml
  python -m master_scm.runner run --config MasterSCM/configs/main.yaml
  ```

- TabPFN and XGBoost baselines:

  ```bash
  python -m master_scm.tabpfn_runner smoke --config MasterSCM/configs/main.yaml
  python -m master_scm.tabpfn_runner run --config MasterSCM/configs/main.yaml

  python -m master_scm.remaining_baselines_runner smoke --config MasterSCM/configs/main.yaml
  python -m master_scm.remaining_baselines_runner run --config MasterSCM/configs/main.yaml
  ```

- Continuous-mediator front-door plug-in estimators:

  ```bash
  python -m master_scm.estimator \
    --methods fd_xgboost fd_nn tabpfn_fd \
    --worlds {0..39} --gpu-ids 0 1 2 3 4 5 6 7
  ```

- P-Learner final-stage variants:

  ```bash
  python -m master_scm.p_learner_runner run \
    --track main --workers 8 --gpu-ids 0,1,2,3,4,5,6,7

  python -m master_scm.p_learner_runner audit --track main
  ```

- Each completed model/world directory includes the prediction, metric,
  configuration, input-contract, seed, and executed-command metadata.

## Reproduce IHDP/IHDP-HC

```bash
python Latent/src/run_ihdp_experiment.py \
  --profile main --num-trials 100 \
  --conditions full hidden --models all --device auto

python -m Latent.src.evaluation.aggregate_ihdp \
  --profile main --num-trials 100

python -m Latent.src.evaluation.verify_experiment \
  --profile main --num-trials 100
```

## Scope

- Figure reproduction uses the saved prediction artifacts and does not retrain
  models.
- Full model reruns require the pretrained checkpoints referenced under
  `MasterSCM/models/`; foundation-model pretraining is not included.
- Superseded experiments are retained under `Past/` and are not part of the
  current reproduction pipeline.
