from keras import backend as K
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D


def CSA(concat1, concat2, kernel_size=7):

    if K.image_data_format() == "channels_first":
        channel1 = concat1.shape[1]
        sa1 = Permute((2, 3, 1))(concat1)
        channel2 = concat2.shape[1]
        sa2 = Permute((2, 3, 1))(concat2)
    else:
        channel1 = concat1.shape[-1]
        sa1 = concat1
        channel2 = concat2.shape[-1]
        sa2 = concat2

    avg_pool1 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(sa1)
    assert avg_pool1.shape[-1] == 1
    avg_pool2 = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(sa2)
    assert avg_pool2.shape[-1] == 1

    sa = Concatenate(axis=3)([avg_pool1, avg_pool2])
    assert sa.shape[-1] == 2

    sa = Conv2D(filters=1,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                use_bias=False)(sa)  
    assert sa.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        sa = Permute((3, 1, 2))(sa)

    return multiply([concat1, sa])
