from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

class QNorm(Layer):
    def __init__(self, initial_scale=1, **kwargs):
        super(QNorm, self).__init__(**kwargs)
        self.shift = None
        self.scale = None
        self.initial_scale = initial_scale

    def build(self, input_shape):
        self.shift = self.add_weight(shape=1,
                                     initializer=initializers.get('zeros'),
                                     trainable=False,
                                     name='{}_shift'.format(self.name))
        self.scale = self.add_weight(shape=1,
                                     initializer=initializers.Constant(self.initial_scale),
                                     trainable=False,
                                     name='{}_scale'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        return (x + self.shift) * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape
