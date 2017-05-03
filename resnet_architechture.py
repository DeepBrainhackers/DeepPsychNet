from keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Dense, Input, Add, Flatten
from keras.models import Model
from keras.optimizers import Adam
from deepPsychNet_keras import balanced_accuracy
from multi_gpu import make_parallel


def conv_block(x, num_conv, base_name, kernel_size=(3, 3, 3)):
    x = BatchNormalization(name='{}-bn1'.format(base_name))(x)
    x = Activation('relu', name='{}-relu1'.format(base_name))(x)

    x = Conv3D(filters=num_conv, kernel_size=kernel_size, padding='same', strides=(1, 1, 1),
               name='{}-conv1'.format(base_name))(x)
    x = BatchNormalization(name='{}-bn2'.format(base_name))(x)
    x = Activation('relu', name='{}-relu2'.format(base_name))(x)

    x = Conv3D(filters=num_conv, kernel_size=kernel_size, padding='same', strides=(1, 1, 1),
               name='{}-conv2'.format(base_name))(x)
    return x


def conv_bn_act(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same', base_name='conv-layer',
                num_layer=1):
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
               name='{}-conv{}'.format(base_name, num_layer))(x)
    x = BatchNormalization(name='{}-bn{}'.format(base_name, num_layer))(x)
    x = Activation('relu', name='{}-relu{}'.format(base_name, num_layer))(x)
    return x


def bn_act(x, base_name='bn_act', num_layer=1):
    x = BatchNormalization(name='{}-bn{}'.format(base_name, num_layer))(x)
    x = Activation('relu', name='{}-relu{}'.format(base_name, num_layer))(x)
    return x


def ResNet(kernel_size=(7, 7, 7)):
    input_layer = Input(batch_shape=(None, 91, 109, 91, 1))

    n_filters_conv = 64
    x = conv_bn_act(input_layer, filters=n_filters_conv, kernel_size=kernel_size, padding='valid', base_name='1st',
                    num_layer=1)
    x = conv_bn_act(x, filters=n_filters_conv, kernel_size=kernel_size, padding='valid', base_name='2nd',
                    num_layer=2)

    n_filters_block = 32
    x = Conv3D(filters=n_filters_block, kernel_size=kernel_size, strides=(2, 2, 2), padding='same', name='3rd-conv3')(x)
    # 1st block of 2 convolutions/batch-norm/relu followed by addiing of outputs and another block
    block = conv_block(x, n_filters_block, 'block1', kernel_size=kernel_size)
    add1 = Add(name='add1')([x, block])
    block2 = conv_block(add1, n_filters_block, 'block2', kernel_size=kernel_size)
    add2 = Add(name='add2')([add1, block2])
    x = bn_act(add2, base_name='bn-act', num_layer=1)

    n_filters_block = 32
    x = Conv3D(filters=n_filters_block, kernel_size=kernel_size, strides=kernel_size, padding='same', name='2nd-conv')(x)
    # 2nd block of 2 convolutions/batch-norm/relu followed by addiing of outputs and another block
    block3 = conv_block(x, n_filters_block, 'block3', kernel_size=kernel_size)
    add3 = Add(name='add3')([x, block3])
    block4 = conv_block(add3, n_filters_block, 'block4', kernel_size=kernel_size)
    add4 = Add(name='add4')([add3, block4])
    x = bn_act(add4, base_name='bn-act', num_layer=2)

    n_filters_block = 32
    x = Conv3D(filters=n_filters_block, kernel_size=kernel_size, strides=(2, 2, 2), padding='same', name='3rd-conv')(x)
    # 3rd block of 2 convolutions/batch-norm/relu followed by addiing of outputs and another block
    block5 = conv_block(x, n_filters_block, 'block5', kernel_size=kernel_size)
    add5 = Add(name='add5')([x, block5])
    block6 = conv_block(add5, n_filters_block, 'block6', kernel_size=kernel_size)
    add6 = Add(name='add6')([add5, block6])

    # Finalizing the network
    x = MaxPool3D(pool_size=(2, 2, 2), name='mp3d1')(add6)
    x = Flatten(name='flatten')(x)
    x = Dense(units=128, activation='relu', name='fc1')(x)
    output = Dense(units=2, activation='softmax')(x)

    model = Model(input_layer, output)
    model.summary()

    model = make_parallel(model, 2)
    model.summary()

    model.compile(optimizer=Adam(lr=1e-6), loss='categorical_crossentropy', metrics=['accuracy', balanced_accuracy])
    return model


if __name__ == '__main__':
    resnet = ResNet()
