import os
import shutil
import tqdm
from typing import Tuple, Optional

LABELS_DIR: str = "labels"
IMAGES_DIR: str = "images"

DEFAULT_DATASET_SPLITS = ("train", "validate", "test")


def copy_dataset(
    src_dir: str,
    dst_dir: str,
    split_id: int = 1,
    dataset_splits: Tuple[str] = DEFAULT_DATASET_SPLITS,
    images_dir: Optional[str] = IMAGES_DIR,
    labels_dir: str = LABELS_DIR,
) -> None:
    src_images_dir = (
        os.path.join(src_dir, images_dir)
        if images_dir is not None
        else src_dir
    )

    dst_split_dir = f"{dst_dir}_{split_id}"
    for split in dataset_splits:
        file_list = os.path.join(src_dir, labels_dir, f"{split}{split_id}.txt")
        copy_subset(src_images_dir, dst_split_dir, split, file_list)


def copy_subset(
    src_images_dir: str, dst_split_dir: str, split: str, file_list: str
) -> None:
    with open(file_list, "r") as f:
        fnames = [line.strip() for line in f.readlines()]
        for fn in tqdm.tqdm(fnames):
            src_fname = os.path.join(src_images_dir, fn)
            dst_fname = os.path.join(dst_split_dir, split, fn)
            os.makedirs(os.path.dirname(dst_fname), exist_ok=True)
            shutil.copy(src_fname, dst_fname)
