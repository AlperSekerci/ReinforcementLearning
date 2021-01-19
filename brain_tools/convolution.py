def conv_layer(input, filters, kernel_size, strides, activate=True):
    out = Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)(input)
    out = BatchNormalization()(out)
    if activate: out = Activation('relu')(out)
    return out

def residual_block(input, filters, kernel_size, strides):
    out = conv_layer(input, filters, kernel_size, strides, activate=True)
    out = conv_layer(out, filters, kernel_size, strides, activate=False)
    out = input + out
    out = Activation('relu')(out)
    return out

def set_mem_growth():
    from tensorflow import config
    gpus = config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: config.experimental.set_memory_growth(gpu, True)
            logical_gpus = config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e: print(e)
