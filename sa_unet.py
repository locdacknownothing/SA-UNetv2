
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Activation, concatenate
from keras_cv.layers import DropBlock2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow_addons.layers import GroupNormalization

from spatial_attention import spatial_attention
from csa import CSA


def SA_UNet(input_size, block_size=7, rate=0.1, start_neurons=16):
    inputs = Input(input_size)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = DropBlock2D(block_size=block_size, rate=rate)(conv1)
    conv1= BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(conv1)
    conv1 = DropBlock2D(block_size=block_size, rate=rate)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)


    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = DropBlock2D(block_size=block_size, rate=rate)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(conv2)
    conv2 = DropBlock2D(block_size=block_size, rate=rate)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)


    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = DropBlock2D(block_size=block_size, rate=rate)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(conv3)
    conv3 = DropBlock2D(block_size=block_size, rate=rate)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)


    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    convm = DropBlock2D(block_size=block_size, rate=rate)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)
    convm = spatial_attention(convm)
    convm = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(convm)
    convm = DropBlock2D(block_size=block_size, rate=rate)(convm)
    convm = BatchNormalization()(convm)
    convm = Activation('relu')(convm)


    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])

    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, rate=rate)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, rate=rate)(uconv3)
    uconv3 = BatchNormalization()(uconv3)
    uconv3 = Activation('relu')(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])

    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, rate=rate)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, rate=rate)(uconv2)
    uconv2 = BatchNormalization()(uconv2)
    uconv2 = Activation('relu')(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])


    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, rate=rate)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, rate=rate)(uconv1)
    uconv1 = BatchNormalization()(uconv1)
    uconv1 = Activation('relu')(uconv1)
    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs, output_layer)

    return model


def SA_UNetGN_SiLU(input_size, block_size=7, rate=0.1, start_neurons=16):
    inputs = Input(input_size)

    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(inputs)
    conv1 = DropBlock2D(block_size=block_size, rate=rate)(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation('silu')(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(conv1)
    conv1 = DropBlock2D(block_size=block_size, rate=rate)(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation('silu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(pool1)
    conv2 = DropBlock2D(block_size=block_size, rate=rate)(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation('silu')(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(conv2)
    conv2 = DropBlock2D(block_size=block_size, rate=rate)(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation('silu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(pool2)
    conv3 = DropBlock2D(block_size=block_size, rate=rate)(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation('silu')(conv3)
    conv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(conv3)
    conv3 = DropBlock2D(block_size=block_size, rate=rate)(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation('silu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = Conv2D(start_neurons * 8, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(pool3)
    convm = DropBlock2D(block_size=block_size, rate=rate)(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation('silu')(convm)
    convm = spatial_attention(convm)
    convm = Conv2D(start_neurons * 8, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(convm)
    convm = DropBlock2D(block_size=block_size, rate=rate)(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation('silu')(convm)

    deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, rate=rate)(uconv3)
    uconv3 = GroupNormalization(groups=8)(uconv3)
    uconv3 = Activation('silu')(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, rate=rate)(uconv3)
    uconv3 = GroupNormalization(groups=8)(uconv3)
    uconv3 = Activation('silu')(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, rate=rate)(uconv2)
    uconv2 = GroupNormalization(groups=8)(uconv2)
    uconv2 = Activation('silu')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, rate=rate)(uconv2)
    uconv2 = GroupNormalization(groups=8)(uconv2)
    uconv2 = Activation('silu')(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same", kernel_initializer='he_normal')(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, rate=rate)(uconv1)
    uconv1 = GroupNormalization(groups=8)(uconv1)
    uconv1 = Activation('silu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), padding="same", activation=None, kernel_initializer='he_normal')(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, rate=rate)(uconv1)
    uconv1 = GroupNormalization(groups=8)(uconv1)
    uconv1 = Activation('silu')(uconv1)

    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None, kernel_initializer='he_normal')(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs, output_layer)
    return model


# 定义 SA_UNetV2 模型
def SA_UNetV2(input_size, block_size=7, rate=0.1, start_neurons=16):
    inputs = Input(input_size)

    # 编码器部分
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(inputs)
    conv1 = DropBlock2D(block_size=block_size, rate=rate)(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation('silu')(conv1)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(conv1)
    conv1 = DropBlock2D(block_size=block_size, rate=rate)(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation('silu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool1)
    conv2 = DropBlock2D(block_size=block_size, rate=rate)(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation('silu')(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(conv2)
    conv2 = DropBlock2D(block_size=block_size, rate=rate)(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation('silu')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(start_neurons * 3, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool2)
    conv3 = DropBlock2D(block_size=block_size, rate=rate)(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation('silu')(conv3)
    conv3 = Conv2D(start_neurons * 3, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(conv3)
    conv3 = DropBlock2D(block_size=block_size, rate=rate)(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation('silu')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # 中间层
    convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(pool3)
    convm = DropBlock2D(block_size=block_size, rate=rate)(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation('silu')(convm)
    convm = spatial_attention(convm)
    convm = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(convm)
    convm = DropBlock2D(block_size=block_size, rate=rate)(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation('silu')(convm)

    deconv3 = Conv2DTranspose(start_neurons * 3, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, CSA(conv3,deconv3)])
    uconv3 = Conv2D(start_neurons * 3, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, rate=rate)(uconv3)
    uconv3 = GroupNormalization(groups=8)(uconv3)
    uconv3 = Activation('silu')(uconv3)
    uconv3 = Conv2D(start_neurons * 3, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv3)
    uconv3 = DropBlock2D(block_size=block_size, rate=rate)(uconv3)
    uconv3 = GroupNormalization(groups=8)(uconv3)
    uconv3 = Activation('silu')(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, CSA(conv2,deconv2)])
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, rate=rate)(uconv2)
    uconv2 = GroupNormalization(groups=8)(uconv2)
    uconv2 = Activation('silu')(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv2)
    uconv2 = DropBlock2D(block_size=block_size, rate=rate)(uconv2)
    uconv2 = GroupNormalization(groups=8)(uconv2)
    uconv2 = Activation('silu')(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, CSA(conv1,deconv1)])
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, rate=rate)(uconv1)
    uconv1 = GroupNormalization(groups=8)(uconv1)
    uconv1 = Activation('silu')(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same",kernel_initializer='he_normal')(uconv1)
    uconv1 = DropBlock2D(block_size=block_size, rate=rate)(uconv1)
    uconv1 = GroupNormalization(groups=8)(uconv1)
    uconv1 = Activation('silu')(uconv1)


    output_layer_noActi = Conv2D(1, (1, 1), padding="same", activation=None,kernel_initializer='he_normal')(uconv1)
    output_layer = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs, output_layer)

    return model
