#

import os
import shutil
import random
from tqdm import tqdm

FMD_SRC_ROOT: str = "/data/fmd"
FMD_IMAGE_DIR: str = "image"
FMD_SPLITS_DIR: str = "splits"
TRAIN_FNAME: str = "train_01.txt"
TEST_FNAME: str = "test_01.txt"
FMD_SPLIT_01: str = "/data/fmd_split_01"

RANDOM_SEED: int = 42


def get_fmd_classes(fmd_src_dir: str) -> list[str]:
    return os.listdir(fmd_src_dir)


def generate_train_test_split(
    fmd_src_dir: str, train_split: float = 0.5
) -> tuple[list[str], list[str]]:
    classes = get_fmd_classes(fmd_src_dir)

    random.seed(RANDOM_SEED)
    train_fnames, test_fnames = [], []

    for class_name in classes:
        fnames = [
            x
            for x in os.listdir(os.path.join(fmd_src_dir, class_name))
            if x.endswith(".jpg")
        ]
        random.shuffle(fnames)

        train_fnames += [
            os.path.join(class_name, x)
            for x in fnames[: int(len(fnames) * train_split)]
        ]
        test_fnames += [
            os.path.join(class_name, x)
            for x in fnames[int(len(fnames) * train_split) :]
        ]

    return train_fnames, test_fnames


def save_filelists(train_fnames, test_fnames):
    for out_fname, img_list in zip(
        [TRAIN_FNAME, TEST_FNAME], [train_fnames, test_fnames]
    ):
        os.makedirs(os.path.join(FMD_SRC_ROOT, FMD_SPLITS_DIR), exist_ok=True)
        with open(os.path.join(FMD_SRC_ROOT, FMD_SPLITS_DIR, out_fname), "w") as f:
            f.write("\n".join(img_list))


def split_fmd_dataset(src_dir: str, dst_dir: str, overwrite: bool = False):
    if overwrite or not os.path.exists(
        os.path.join(src_dir, FMD_SPLITS_DIR, TRAIN_FNAME)
    ):
        train_list, test_list = generate_train_test_split(
            os.path.join(src_dir, FMD_IMAGE_DIR)
        )
        save_filelists(train_list, test_list)

    if os.path.exists(dst_dir):
        if overwrite:
            shutil.rmtree(dst_dir)
        else:
            return

    for split_fname in tqdm((TRAIN_FNAME, TEST_FNAME)):
        with open(os.path.join(src_dir, FMD_SPLITS_DIR, split_fname), "r") as f:
            fnames = f.read().splitlines()
        # TODO: better way to handle this
        split_dir = split_fname.split("_")[0]
        for fname in tqdm(fnames):
            src_fname = os.path.join(src_dir, FMD_IMAGE_DIR, fname)
            dst_fname = os.path.join(dst_dir, split_dir, fname)
            os.makedirs(os.path.dirname(dst_fname), exist_ok=True)
            shutil.copy2(src_fname, dst_fname)


if __name__ == "__main__":
    split_fmd_dataset(FMD_SRC_ROOT, FMD_SPLIT_01, overwrite=True)
