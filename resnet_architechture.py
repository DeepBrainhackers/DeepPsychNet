from keras.layers import Conv3D, BatchNormalization, Activation, MaxPool3D, Dense, Input, Add, Flatten
from keras.models import Model
from deepPsychNet_keras import balanced_accuracy


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


def ResNet():
    input_layer = Input(batch_shape=(None, 91, 109, 91, 1))

    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), name='init-conv1')(input_layer)
    x = BatchNormalization(name='init-bn1')(x)
    x = Activation('relu', name='init-relu1')(x)

    x = Conv3D(filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), name='init-conv2')(x)
    x = BatchNormalization(name='init-bn2')(x)
    x = Activation('relu', name='init-relu2')(x)

    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), name='init-conv3')(x)
    block = conv_block(x, 64, 'block1')

    add1 = Add(name='add1')([x, block])

    block2 = conv_block(add1, 64, 'block2')
    add2 = Add(name='add2')([add1, block2])

    x = BatchNormalization(name='2nd-bn1')(add2)
    x = Activation('relu', name='2nd-relu1')(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), strides=(2, 2, 2), name='2nd-conv')(x)

    block3 = conv_block(x, 64, 'block3')
    add3 = Add(name='add3')([x, block3])

    block4 = conv_block(add3, 64, 'block4')
    add4 = Add(name='add4')([add3, block4])

    x = BatchNormalization(name='3rd-bn1')(add4)
    x = Activation('relu', name='3rd-relu1')(x)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), strides=(2, 2, 2), name='3rd-conv')(x)

    block5 = conv_block(x, 128, 'block5')
    add5 = Add(name='add5')([x, block5])

    block6 = conv_block(add5, 128, 'block6')
    add6 = Add(name='add6')([add5, block6])

    x = MaxPool3D(pool_size=(7, 7, 7))(add6)
    x = Flatten(name='flatten')(x)
    x = Dense(units=128, activation='relu')(x)
    output = Dense(units=2, activation='softmax')(x)

    model = Model(input_layer, output)

    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', balanced_accuracy])
    return model


if __name__ == '__main__':
    resnet = ResNet()
