from easydict import EasyDict

AUGMENTATION_PROMPT_PTH = "constants/augmentations.txt"

DEFAULT_LDM_MODEL_KEY = "stable-v1-4"
DEFAULT_BATCH_SIZE = 6

MODELS = {
    "stable-v1-4": dict(
        config="imports/ldm/configs/stable-diffusion/v1-inference.yaml", 
        ckpt="imports/ldm/models/ldm/stable-diffusion/sd-v1-4.ckpt",
        plms=True
    )
}

# Classifier-Guidance Scale: 7.5 / Image Size: 512 / PLMS Sampling: On
def GetLDMOptions(model_key: str, models: dict):
    # Basic Options
    opt = EasyDict()
    opt.H = 512
    opt.W = 512
    opt.C = 4
    opt.f = 8
    opt.ddim_steps = 50
    opt.fixed_code = False
    opt.ddim_eta = 0.0
    # opt.seed = 42
    opt.scale = 7.5
    opt.precision = "autocast"
    opt.model_type = model_key
    opt.plms = models[opt.model_type]["plms"]
    opt.config = models[opt.model_type]["config"]
    opt.ckpt = models[opt.model_type]["ckpt"]
    if "C" in list(models[opt.model_type].keys()):
        print("C changed.")
        opt.C = models[opt.model_type]["C"]
    return opt