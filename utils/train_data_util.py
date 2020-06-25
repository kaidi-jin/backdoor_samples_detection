import numpy  as np
import h5py
from keras.utils import np_utils
from keras.datasets import mnist
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def load_h5_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset

def load_mnist_dataset():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    # input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    X_train = x_train
    Y_train = y_train
    X_test = x_test
    Y_test = y_test

    return X_train, Y_train, X_test, Y_test


def load_gtsrb_dataset(data_file='../data/gtsrb/gtsrb_dataset.h5'):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        exit(1)

    dataset = load_h5_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    return X_train, Y_train, X_test, Y_test


def load_train_dataset(DATASET='mnist'):
    if DATASET == 'mnist':
        return load_mnist_dataset()
    else:
        return load_gtsrb_dataset()
