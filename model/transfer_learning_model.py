import tensorflow as tf
from common.constants import DatasetConstants


def feature_extractor(inputs):

    feature_extraction = tf.keras.applications.resnet.ResNet50(
        input_shape=(DatasetConstants.HEIGHT,
                     DatasetConstants.WIDTH,
                     3),
        include_top=False, weights="imagenet")(inputs)
    return feature_extraction


def classifier(inputs):

    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024, activation="relu")(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dense(10, activation="softmax",
                              name="classification")(x)
    return x


def full_model(inputs):

    resnet_feat_extractor = feature_extractor(inputs)
    classification_output = classifier(resnet_feat_extractor)
    return classification_output


def compile_model():

    inputs = tf.keras.layers.Input(shape=(DatasetConstants.HEIGHT,
                                          DatasetConstants.WIDTH, 3))
    classification_output = full_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=classification_output)
    model.compile(optimizer="SGD", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
