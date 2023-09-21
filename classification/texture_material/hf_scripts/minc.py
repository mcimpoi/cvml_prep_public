#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Script to copy MINC dataset to Hugging Face """
import os
import datasets
import glob
from tqdm import tqdm

from classification.texture_material.hf_scripts.common import copy_dataset

SRC_DATA_DIR: str = "/data/minc-2500"
DST_DATA_DIR: str = "/data/hf/minc-2500_split"

SPLIT_ID: int = 1
HF_HUB_MINC_DATASET_NAME: str = f"mcimpoi/minc-2500_split_{SPLIT_ID}"


def patch_minc_2500(dataset_root: str) -> None:
    """Ensure the minc-2500 dataset follows the structure as expected by
    the script"""

    for validation_fname in glob.glob(f"{dataset_root}/labels/valid*.txt"):
        os.rename(
            validation_fname, validation_fname.replace("validate", "valid")
        )

    for split_fname in tqdm(glob.glob(f"{dataset_root}/labels/*.txt")):
        with open(split_fname, "r") as f:
            lines = f.readlines()
        with open(split_fname, "w") as f:
            for line in lines:
                f.write(line.replace("images/", ""))


if __name__ == "__main__":
    try:
        dataset = datasets.load_dataset(HF_HUB_MINC_DATASET_NAME)
    except FileNotFoundError:
        patch_minc_2500(SRC_DATA_DIR)

        dst_split_dir = f"{DST_DATA_DIR}_{SPLIT_ID}"

        if not os.path.exists(dst_split_dir):
            os.makedirs(dst_split_dir, exist_ok=True)
            copy_dataset(
                SRC_DATA_DIR,
                DST_DATA_DIR,
                dataset_splits=("train", "valid", "test"),
            )
        dataset = datasets.load_dataset(
            name=HF_HUB_MINC_DATASET_NAME,
            path=dst_split_dir,
        )
        dataset.push_to_hub(HF_HUB_MINC_DATASET_NAME)

    print(dataset)
