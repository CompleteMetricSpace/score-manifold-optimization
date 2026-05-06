import yaml
from diffusion.data import load_dataset
from diffusion.training import create_diffusion
from diffusion.models import create_model

with open("../configs/train_rizon4_quick.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset = load_dataset("../data/rizon4_T100_n20000.pt", device="cpu", reconstruct_constraint=False)
space = dataset.get_space()
diffusion = create_diffusion(space, cfg)

model = create_model(
    model_type=str(cfg["model"]["type"]),
    space=space,
    diffusion=diffusion,
    config=cfg,
    initialize=False,
)

print(model.get_num_parameters())
