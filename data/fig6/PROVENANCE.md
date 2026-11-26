# Figure 6 data provenance

The inputs and numerical references come from `causalidview_39_reproduce`. `archive_manifest.json` records the original archive paths and SHA-256 hashes alongside the hashes of these compact release files. The scoring and bootstrap protocol in `scripts/fig6/verify_fig6.py` is adapted from the archive's `fig6.py` and `common.py`.

| Release input | Source and contents |
| --- | --- |
| `synthetic_targets.csv.gz` | Held-out query IDs and `tau` from the 40 MasterSCM world `potentials.npz` files |
| `synthetic_predictions.csv.gz` | Back-door and hidden-confounding estimates, query IDs, and recorded targets for six estimators and 40 worlds. The three CFM hidden-confounding predictions were identical at all six Γ levels and are stored once per query. |
| `ihdp_predictions.csv.gz` | Archived test-set estimates and recorded targets for six estimators, all 100 trials, and both conditions. `data/ihdp_hc/trial_*/evaluator_only.npz` supplies test IDs and targets, which match the corresponding arrays from the archive. |
| `bound_targets.csv.gz` | Held-out query IDs and `tau0` from the 40 partial-identification world files |
| `bound_oracles.csv.gz` | Oracle lower and upper endpoints for Manski, IV, and the six sensitivity Γ levels |
| `bound_predictions.csv.gz` | Saved predicted endpoints from the `full` and `tabpfn_b_learner` outputs. Lower/upper ordering is preserved as recorded. |

The default scoring command uses the held-out targets, the oracle endpoints, and new prediction files. The three saved prediction files above are used by the optional `--archived` command. For sensitivity, both commands select the Γ=2 rows from CSA-PFN, NeuralCSA, and B-Learner + TabPFN (all) for panel (c).

The default comparison uses the bundled numerical references from the `causalidview_39_reproduce` archive. The combined verifier's optional `--paper-reference` check uses the 18 bar heights in `results/fig6/paper_bar_values.csv`, which were read from the arXiv v1 PDF during the `causalidview_70_reproduce` handoff. The manifest records the source file hash.

Panels (a,b) display 95% bootstrap intervals for mean sPEHE across worlds or trials. Panel (c) displays the mean and sample standard deviation of endpoint RMSE across 40 worlds. The sensitivity reference in `results/fig6/panel_c.csv` uses Γ=2, the setting specified in the v1 caption. The v2 per-world reference records the corresponding selection and original file hash in `archive_manifest.json`.
