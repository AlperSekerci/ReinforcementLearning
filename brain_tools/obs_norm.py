from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

class ObsNormalizer(Layer):
    def __init__(self, **kwargs):
        super(ObsNormalizer, self).__init__(**kwargs)
        self.shift = None
        self.scale = None

    def build(self, input_shape):
        shape = input_shape[1:]
        self.shift = self.add_weight(shape=shape,
                                     initializer=initializers.get('zeros'),
                                     trainable=False,
                                     name='{}_shift'.format(self.name))
        self.scale = self.add_weight(shape=shape,
                                     initializer=initializers.get('ones'),
                                     trainable=False,
                                     name='{}_scale'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        return (x + self.shift) * self.scale

    def compute_output_shape(self, input_shape):
        return input_shape
