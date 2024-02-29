from dataclasses import asdict
from datetime import datetime

import glob
import json
import logging
import os
from pathlib import Path

from segmenters import ClusterSegmenter, SAMSegmenter
from config import Config, cfg
from models import (
    AMGSettings,
    BBox,
    Device,
    MaskFormat,
    ModelType,
)


THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

log_path = THIS_DIR / f"../logs/segmentation_dataset_{cfg.DATASET_NAME}_{now_str}.log"

# Send logs to a file and stdout
logging.basicConfig(
    # filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=cfg.LOG_LEVEL.upper(),
    handlers=[
        logging.FileHandler(filename=log_path.resolve(), mode="w"),
        logging.StreamHandler(),
    ],
)


def segment_dataset(config: Config) -> None:
    """
    Segments the dataset using the specified configuration settings.

    This function reads input files from the input directory, applies
    segmentation using the SAMSegmenter,and saves the segmentation
    results as JSON files in the output directory.

    Returns:
        None
    """

    input_dir = THIS_DIR / ".." / config.INPUT_DIR / config.DATASET_NAME
    input_dir = input_dir.resolve()

    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} not found")

    ROI = BBox(
        x_min=config.X_MIN,
        y_min=config.Y_MIN,
        width=config.X_MAX - config.X_MIN,
        height=config.Y_MAX - config.Y_MIN,
    )

    amg_settings = AMGSettings(
        points_per_side=config.POINTS_PER_SIDE,
        points_per_batch=config.POINTS_PER_BATCH,
    )

    sam = SAMSegmenter(
        model_type=ModelType(config.MODEL_TYPE),
        checkpoint=config.CHECKPOINT,
        device=Device(config.DEVICE),
        mask_format=MaskFormat(config.MASK_FORMAT),
        amg_settings=amg_settings,
    )

    segmenter = ClusterSegmenter(sam)

    logging.info(f"Model Settings: \n{amg_settings.to_json()}")
    logging.info(f"Using model: {sam.metadata.name} {sam.metadata.version}")

    input_files = [
        Path(f) for f in glob.glob(f"{input_dir}/**/*.{config.IMG_EXT}", recursive=True)
    ]

    logging.info(f"Found {len(input_files)} input files")

    begin = datetime.now()

    logging.info("Starting Segmentation ...")

    for input_file in input_files:
        try:
            logging.info(f"Processing {input_file} ...")

            results = segmenter.generate(input_file, bbox=ROI, dry_run=config.DRY_RUN)

            logging.info(f"Found {len(results.masks)} masks")

            output_path = Path(str(input_file).replace("input", "output")).with_suffix(
                ".json"
            )

            logging.info(f"Saving results to {output_path}")

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(asdict(results), f, indent=4)
        except Exception as e:
            logging.error(f"Error processing {input_file}: {e}")

    end = datetime.now()

    logging.info("Segmentation Finished")

    logging.info(f"Total time: {end - begin}")


if __name__ == "__main__":
    segment_dataset(config=cfg)
