import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from common.constants import DatasetConstants


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    img = tf.keras.utils.load_img(path,
                                  target_size=(DatasetConstants.HEIGHT,
                                               DatasetConstants.WIDTH))

    return tf.keras.utils.img_to_array(img)


def display_all_classes():

    fig, ax = plt.subplots(2, 5, figsize=(30, 15))
    ax = ax.ravel()

    i = 0
    for cls in DatasetConstants.CLASSES:
        path = DatasetConstants.FULL_TRAINING_DIR + f"/{cls}"
        img_path = random.sample(os.listdir(path), 1)[0]
        img = load_image_into_numpy_array(path+"/"+img_path)
        ax[i].set_title(cls, fontsize=20)
        ax[i].imshow(img.astype('uint8'))
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
        i += 1
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_all_classes()
