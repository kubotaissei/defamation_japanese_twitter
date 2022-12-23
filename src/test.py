import hydra
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer
import pandas as pd


@hydra.main(version_base=None, config_path="config", config_name="baseline")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print(cfg.data.train_path)
    # print
    # pd.read_csv(data.trani)


if __name__ == "__main__":
    my_app()
