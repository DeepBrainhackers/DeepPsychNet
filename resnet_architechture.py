from keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Dense, Input, Add, Flatten
from keras.models import Model
from deepPsychNet_keras import balanced_accuracy
from multi_gpu import make_parallel


def conv_block(x, num_conv, base_name):
    x = BatchNormalization(name='{}-bn1'.format(base_name))(x)
    x = Activation('relu', name='{}-relu1'.format(base_name))(x)

    x = Conv3D(filters=num_conv, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),
               name='{}-conv1'.format(base_name))(x)
    x = BatchNormalization(name='{}-bn2'.format(base_name))(x)
    x = Activation('relu', name='{}-relu2'.format(base_name))(x)

    x = Conv3D(filters=num_conv, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1),
               name='{}-conv2'.format(base_name))(x)
    return x


def conv_bn_act(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), base_name='conv-layer', num_layer=1):
    x = Conv3D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
               name='{}-conv{}'.format(base_name, num_layer))(x)
    x = BatchNormalization(name='{}-bn{}'.format(base_name, num_layer))(x)
    x = Activation('relu', name='{}-relu{}'.format(base_name, num_layer))(x)
    return x


def bn_act(x, base_name='bn_act', num_layer=1):
    x = BatchNormalization(name='{}-bn{}'.format(base_name, num_layer))(x)
    x = Activation('relu', name='{}-relu{}'.format(base_name, num_layer))(x)
    return x


def ResNet():
    input_layer = Input(batch_shape=(None, 91, 109, 91, 1))

    x = conv_bn_act(input_layer, filters=32, base_name='1st', num_layer=1)
    x = conv_bn_act(x, filters=32, base_name='2nd', num_layer=2)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', name='3rd-conv3')(x)
    # 1st block of 2 convolutions/batch-norm/relu followed by addiing of outputs and another block
    block = conv_block(x, 64, 'block1')
    add1 = Add(name='add1')([x, block])
    block2 = conv_block(add1, 64, 'block2')
    add2 = Add(name='add2')([add1, block2])
    x = bn_act(add2, base_name='bn-act', num_layer=1)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', name='2nd-conv')(x)
    # 2nd block of 2 convolutions/batch-norm/relu followed by addiing of outputs and another block
    block3 = conv_block(x, 64, 'block3')
    add3 = Add(name='add3')([x, block3])
    block4 = conv_block(add3, 64, 'block4')
    add4 = Add(name='add4')([add3, block4])
    x = bn_act(add4, base_name='bn-act', num_layer=2)

    x = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same', name='3rd-conv')(x)
    # 3rd block of 2 convolutions/batch-norm/relu followed by addiing of outputs and another block
    block5 = conv_block(x, 128, 'block5')
    add5 = Add(name='add5')([x, block5])
    block6 = conv_block(add5, 128, 'block6')
    add6 = Add(name='add6')([add5, block6])

    # Finalizing the network
    x = MaxPool3D(pool_size=(7, 7, 7), name='mp3d1')(add6)
    x = Flatten(name='flatten')(x)
    x = Dense(units=128, activation='relu', name='fc1')(x)
    output = Dense(units=2, activation='softmax')(x)

    model = Model(input_layer, output)
    model.summary()

    model = make_parallel(model, 2)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', balanced_accuracy])
    return model


if __name__ == '__main__':
    resnet = ResNet()
