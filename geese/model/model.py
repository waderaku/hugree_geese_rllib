from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from geese.constants import ACTIONLIST


@tf.keras.utils.register_keras_serializable()
class BaseModel(tf.keras.models.Model):
    def __init__(
        self,
        num_layers: int,
        num_filters: int,
        kernel_size: Tuple[int, int],
        bn: bool,
        use_gpu: bool,
    ):
        super().__init__()
        self._num_layers = num_layers
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._bn = bn
        self._use_gpu = use_gpu

        self._init_block = TorusConv2d(
            num_filters=num_filters, kernel_size=kernel_size, bn=bn, use_gpu=use_gpu
        )
        self._blocks = [
            TorusConv2d(
                num_filters=num_filters, kernel_size=kernel_size, bn=bn, use_gpu=use_gpu
            )
            for _ in range(num_layers)
        ]
        self._head_p = tf.keras.layers.Dense(units=len(ACTIONLIST), use_bias=False)
        self._head_v = tf.keras.layers.Dense(units=1, use_bias=False)
        self._flatten = tf.keras.layers.Flatten()

    def call(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        out = self._init_block(x)
        for block in self._blocks:
            out = tf.keras.activations.relu(block(out) + out)
        flatten = self._flatten(out)
        p = self._head_p(flatten)
        v = self._head_v(flatten)
        p = tf.keras.activations.softmax(p)
        v = tf.keras.activations.tanh(v)
        v = tf.squeeze(v, axis=-1)
        return p, v

    def get_config(self) -> dict:
        return {
            "num_layers": self._num_layers,
            "num_filters": self._num_filters,
            "kernel_size": self._kernel_size,
            "bn": self._bn,
            "use_gpu": self._use_gpu,
        }

    @classmethod
    def from_config(cls, config: dict) -> BaseModel:
        parameter = cls(**config)
        return BaseModel(parameter)


class TorusConv2d(tf.keras.layers.Layer):
    def __init__(
        self,
        num_filters: int,
        kernel_size: Tuple[int, int],
        bn: bool,
        use_gpu: bool,
    ):
        super().__init__()
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._bn = bn
        self._use_gpu = use_gpu

        if use_gpu:
            data_format = "channels_first"
        else:
            data_format = "channels_last"

        self._edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self._conv = tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            data_format=data_format,
        )
        self._bn = tf.keras.layers.BatchNormalization() if bn else None

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = tf.concat(
            [x[:, :, :, -self._edge_size[1] :], x, x[:, :, :, : self._edge_size[1]]],
            axis=3,
        )
        out = tf.concat(
            [out[:, :, -self._edge_size[0] :], out, out[:, :, : self._edge_size[0]]],
            axis=2,
        )
        if not self._use_gpu:
            out = tf.transpose(out, [0, 2, 3, 1])
        out = self._conv(out)
        if not self._use_gpu:
            out = tf.transpose(out, [0, 3, 1, 2])
        if self._bn is not None:
            out = self._bn(out)
        return out

    def get_config(self) -> dict:
        return {
            "num_filters": self._num_filters,
            "kernel_size": self._kernel_size,
            "bn": self._bn,
            "use_gpu": self._use_gpu,
        }

    @classmethod
    def from_config(cls, config: dict) -> TorusConv2d:
        parameter = cls(**config)
        return TorusConv2d(parameter)
