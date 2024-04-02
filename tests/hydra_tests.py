import hydra
from hydra.core.config_store import ConfigStore

from gen_names.config import Params, MambaModel, Scheduler_ReduceOnPlateau, Scheduler_OneCycleLR
from gen_names.models.mamba import Mamba

import torch
import lightning as L

cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="model", name="base_mamba", node=MambaModel)
cs.store(group="scheduler", name="base_rop", node=Scheduler_ReduceOnPlateau)
cs.store(group="scheduler", name="base_oclr", node=Scheduler_OneCycleLR)


@hydra.main(config_path="../generating_small_sentences/conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    print(dir(cfg))
    print(cfg)

    L.seed_everything(cfg.training.seed)

    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    model = Mamba(cfg)

    x = torch.LongTensor([
        [12, 3, 16, 8, 1, 15, 20],
        [16, 10, 90, 81, 13, 0, 0]
    ])

    out = model(x)
    print(torch.argmax(out, dim=-1))


if __name__ == "__main__":
    main()