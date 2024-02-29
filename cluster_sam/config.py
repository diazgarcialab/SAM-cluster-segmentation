from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    # Defaults values will be overwritten by variables in .env file
    # Read more about pydantic-settings at
    # https://docs.pydantic.dev/latest/concepts/pydantic_settings/
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')
    LOG_LEVEL: str = 'INFO'
    INPUT_DIR: Path # 'data/input/clusters'
    DATASET_NAME: str # '1_side_test'
    MODEL_TYPE: str # 'vit_h'
    CHECKPOINT: str # 'models/SAM/weights/sam_vit_h_4b8939.pth'
    DEVICE: str # 'cpu'
    POINTS_PER_BATCH: int # 128
    MASK_FORMAT: str # 'coco_rle'
    POINTS_PER_SIDE: int # 32
    DRY_RUN: bool # False
    X_MIN: int # 2000  # NOTE: Defined by Experimental Design (Pixels)
    X_MAX: int # 3600  #       This 4 values define the
    Y_MIN: int # 500   #       Region of Interest (ROI)
    Y_MAX: int # 2700
    IMG_EXT: str # 'JPG'


cfg = Config() # Singleton Configuration Object


if __name__ == "__main__":
    print(f"Config: \n{cfg.model_dump_json(indent=4)}")
