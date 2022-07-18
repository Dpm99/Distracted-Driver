import os
import random
from shutil import copyfile
from common.constants import DatasetConstants


def make_train_val_dirs(source_dir):
    train_name = os.path.join(source_dir, "training")
    val_name = os.path.join(source_dir, "validation")

    os.makedirs(train_name)
    os.makedirs(val_name)

    for c in DatasetConstants.CLASSES:
        os.makedirs(os.path.join(train_name, c))
        os.makedirs(os.path.join(val_name, c))


def split_data(source_dir, train_path, val_path, split_size):
    images = os.listdir(source_dir)[:50]
    shuffle_images = random.sample(images, len(images))
    train_size = int(len(shuffle_images) * split_size)
    train_images = shuffle_images[:train_size]
    val_images = shuffle_images[train_size:]

    _copy_images(train_images, source_dir, train_path)
    _copy_images(val_images, source_dir, val_path)


def _copy_images(images, source_dir, destination):
    for im in images:
        copyfile(os.path.join(source_dir, im),
                 os.path.join(destination, im))
