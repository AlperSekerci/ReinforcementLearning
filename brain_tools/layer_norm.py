from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
#from tensorflow import nn
from tensorflow.keras import backend as K

# Source: https://stackoverflow.com/questions/39095252/fail-to-implement-layer-normalization-with-keras
class LayerNorm(Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """
    def __init__(self, scale_initializer='ones', bias_initializer='zeros', center=True, scale=True, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = 1e-3
        if scale: self.scale_initializer = initializers.get(scale_initializer)
        if center: self.bias_initializer = initializers.get(bias_initializer)
        self.use_bias = center
        self.use_scale = scale
        self.scale = None
        self.bias = None

    def build(self, input_shape):
        if self.use_scale:
            self.scale = self.add_weight(shape=(input_shape[-1],),
                                         initializer=self.scale_initializer,
                                         trainable=True,
                                         name='{}_scale'.format(self.name))
        if self.use_bias:
            self.bias = self.add_weight(shape=(input_shape[-1],),
                                        initializer=self.bias_initializer,
                                        trainable=True,
                                        name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        #mean, variance = nn.moments(x, -1, keepdims=True)
        #return nn.batch_normalization(x, mean, variance, offset=self.bias, scale=self.scale, variance_epsilon=self.epsilon)
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        norm = (x - mean) / (std + self.epsilon)
        if self.use_scale: norm *= self.scale
        if self.use_bias: norm += self.bias
        return norm

    def compute_output_shape(self, input_shape):
        return input_shape
