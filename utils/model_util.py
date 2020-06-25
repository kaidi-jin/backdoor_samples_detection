#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Convolution2D, Conv2D, MaxPooling2D, Input, Dense, Activation, Flatten,Dropout
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.optimizers import Adam,Adadelta
from keras.models import Model
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, Input, Dense, Activation, Flatten,Dropout
from keras.models import Sequential
from keras.optimizers import Adam,Adadelta
# from keras_vggface.vggface import VGGFace

def load_mnist_model():
    # convolution kernel size
    kernel_size = (5, 5)
    # input image dimensions
    input_tensor = Input(shape=(28,28,1))

    # block1
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(10, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)

    opt = Adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def load_gtsrb_model(base=32, dense=512, num_classes=43):
    input_shape = (32, 32, 3)
    model = Sequential()
    model.add(Conv2D(base, (3, 3), padding='same',
                     input_shape=input_shape,
                     activation='relu'))
    model.add(Conv2D(base, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 2, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(base * 4, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(base * 4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(dense, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    opt = Adam(lr=0.001, decay=1 * 10e-5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model



# def vggface_model():
#     hidden_dim = 4096
#     NUM_CLASSES = 83
#     vgg_model = VGGFace(model='vgg16',weights=None,include_top=False, input_shape=(224, 224, 3))
#     last_layer = vgg_model.get_layer('pool5').output
#     x = Flatten(name='flatten')(last_layer)
#     x = Dense(hidden_dim, activation='relu', name='fc6')(x)
#     x = Dense(hidden_dim, activation='relu', name='fc7')(x)
#     out = Dense(NUM_CLASSES, activation='softmax', name='fc8')(x)
#     pubfig_vgg_model = Model(vgg_model.input, out)
#     # compiling
#     lr_optimizer=Adadelta(lr = 0.03)
#     pubfig_vgg_model.compile(loss='categorical_crossentropy', optimizer=lr_optimizer, metrics=['accuracy'])
    
#     return pubfig_vgg_model



def load_keras_model(DATASET='mnist'):
    if DATASET == 'mnist':
        return load_mnist_model()
    else:
        return load_gtsrb_model()