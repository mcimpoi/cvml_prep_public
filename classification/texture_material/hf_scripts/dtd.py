#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Script to copy DTD dataset to Hugging Face """
from classification.texture_material.hf_scripts.common import copy_dataset

SRC_DATA_DIR: str = "/data/dtd"
DST_DATA_DIR: str = "/data/hf/dtd_split"


if __name__ == "__main__":
    copy_dataset(
        SRC_DATA_DIR, DST_DATA_DIR, dataset_splits=("train", "val", "test")
    )
