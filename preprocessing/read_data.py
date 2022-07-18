import tensorflow as tf
import os


def image_generators(train_dir, val_dir, batch_size,
                     class_mode, target_size):

    assert os.path.isdir(train_dir) is True, f"{train_dir} not a directory, please check."
    assert os.path.isdir(val_dir) is True, f"{val_dir} not a directory, please check."

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        shear_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        batch_size=batch_size,
        class_mode=class_mode,
        target_size=target_size
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    val_generator = val_datagen.flow_from_directory(
        directory=val_dir,
        batch_size=batch_size,
        class_mode=class_mode,
        target_size=target_size
    )

    return train_generator, val_generator
