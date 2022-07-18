class DatasetConstants(object):

    DATASET_DIR = "C:/Users/DM3/Project_repositories/DistractedDriver/data/imgs"
    CLASSES = ["c0", "c1", "c2", "c3", "c4", "c5", "c6",
               "c7", "c8", "c9"]
    FULL_TRAINING_DIR = f"{DATASET_DIR}/full_train_data/"
    TRAINING_DIR = f"{DATASET_DIR}/training/"
    VALIDATION_DIR = f"{DATASET_DIR}/validation/"
    TESTING_DIR = f"{DATASET_DIR}/test/"
    HEIGHT = 224
    WIDTH = 224
    BATCH_SIZE = 256


class ModelConstants(object):
    EPOCHS = 4
