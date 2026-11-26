from ..config import BASELINES_ROOT


GAMMAS = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0)
SENSITIVITY_GAMMA = 2.0
SENSITIVITY_GAMMA_INDEX = GAMMAS.index(SENSITIVITY_GAMMA)
B_LEARNER_COMMIT = "734fd7639da54554e41d944bfc2187a24e75cf81"
NEURALCSA_COMMIT = "e8d3f952276379cc2c26ad553a142bdf5083ef52"
ASSET_ROOT = BASELINES_ROOT / "partial_identification"
NEURALCSA_ROOT = ASSET_ROOT / "NeuralCSA"
CSA_PFN_SPEC = {
    "path": "Baselines/partial_identification/checkpoints/csa_pfn_msm/model.pt",
    "source": (
        "https://huggingface.co/arunkatika/csa-pfn-msm/blob/"
        "26466a4128768a134fc96939c9a4191afc83980c/checkpoint/model.pt"
    ),
    "sha256": "f5c67c61056e8348821335688d9df8f695d3dd92d428fc73fc85b0bbe59b5757",
}
