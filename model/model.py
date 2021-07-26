import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="geese", name="visionnet")
class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            strides=1,
            padding="same",
            activation="relu",
        )
        self.conv_value_1 = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=[3, 3],
            strides=1,
            padding="same",
            activation="relu",
        )

        self.conv2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3], strides=1, padding="same", activation="relu"
        )
        self.conv_value_2 = tf.keras.layers.Conv2D(
            filters=32, kernel_size=[3, 3], strides=1, padding="same", activation="relu"
        )

        self.conv3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[7, 11],
            strides=1,
            padding="valid",
            activation="relu",
        )
        self.conv_value_3 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[7, 11],
            strides=1,
            padding="valid",
            activation="relu",
        )

        self.output_predict_layer = tf.keras.layers.Dense(4, activation="softmax")
        self.output_value_layer = tf.keras.layers.Dense(1)

    def call(self, inputs):
        pred = self.conv1(inputs)
        pred = self.conv2(pred)
        pred = self.conv3(pred)
        pred = self.output_predict_layer(pred)

        value = self.conv_value_1(inputs)
        value = self.conv_value_2(value)
        value = self.conv_value_3(value)
        value = self.output_value_layer(value)

        pred = tf.squeeze(pred, (1, 2))
        value = tf.squeeze(value, (1, 2, 3))
        return pred, value
