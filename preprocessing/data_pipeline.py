from common.constants import DatasetConstants
from preprocessing.read_data import image_generators
from preprocessing.train_val_split import make_train_val_dirs, split_data


def run_data_pipeline(split):
    if split:
        print("Splitting data into training and validation.")
        make_train_val_dirs(DatasetConstants.DATASET_DIR)
        for cls in DatasetConstants.CLASSES:
            split_data(DatasetConstants.FULL_TRAINING_DIR+cls,
                       DatasetConstants.TRAINING_DIR+cls,
                       DatasetConstants.VALIDATION_DIR+cls,
                       0.8)
    print("Creating Image Generators.")
    train_gen, val_gen = image_generators(
        DatasetConstants.TRAINING_DIR,
        DatasetConstants.VALIDATION_DIR,
        DatasetConstants.BATCH_SIZE,
        "categorical",
        (DatasetConstants.HEIGHT, DatasetConstants.WIDTH)
    )
    return train_gen, val_gen
