# Figure 6 bound estimators

The estimator code is in [`Baselines/src/baselines/partial_identification/`](../src/baselines/partial_identification/). It preserves the original partial-identification estimator code and experiment settings from CausalArena revision `6d0387f784b508ed240b57ce34d06f06d45a5ab5`. The runner reads frozen observed arrays from [`data/fig6/bound_worlds/`](../../data/fig6/bound_worlds/).

- B-Learner uses the source and license from [CausalML/BLearner at `734fd76`](https://github.com/CausalML/BLearner/tree/734fd7639da54554e41d944bfc2187a24e75cf81). The required modules are bundled under [`_blearner/`](../src/baselines/partial_identification/_blearner/).
- [`prepare_fig6_bounds.py`](../../scripts/fig6/prepare_fig6_bounds.py) fetches [NeuralCSA at `e8d3f95`](https://github.com/DennisFrauen/NeuralCSA/tree/e8d3f952276379cc2c26ad553a142bdf5083ef52) and applies the original [Lightning device compatibility patch](NeuralCSA.patch).
- The same command installs and verifies the [CSA-PFN MSM checkpoint at `26466a4`](https://huggingface.co/arunkatika/csa-pfn-msm/blob/26466a4128768a134fc96939c9a4191afc83980c/checkpoint/model.pt). TabPFN uses the shared [Figure 2 checkpoint setup](../../README.md#model-setup).
