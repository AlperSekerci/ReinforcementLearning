def create_batchnorm_layers(input, norm_layer_count=1, layer_size=128, activation='relu', end_with_norm=False):
    from tensorflow.keras.layers import Dense, BatchNormalization, Activation
    out = input
    for i in range(norm_layer_count):
        out = Dense(layer_size)(out)
        out = BatchNormalization()(out)
        out = Activation(activation)(out)
    if not end_with_norm: out = Dense(layer_size, activation=activation)(out)
    return out