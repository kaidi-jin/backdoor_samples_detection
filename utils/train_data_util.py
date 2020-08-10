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

def load_face_dataset(hdf5_path='../data/pubfig/clean_pubfig_dataset.h5',mean=False):
    if not os.path.exists(hdf5_path):
        print(
            "The data file path %s does not exist."%(hdf5_path) )
        exit(1)
    NUM_CLASSES = 83
    hdf5_file = h5py.File(hdf5_path, "r")
    train_data = hdf5_file["train_img"][:]
    train_label = hdf5_file["train_labels"][:]
    train_label = np_utils.to_categorical(train_label,NUM_CLASSES) 
    test_data = hdf5_file["test_img"][:]
    test_label = hdf5_file["test_labels"][:]
    test_label = np_utils.to_categorical(test_label,NUM_CLASSES) 
    val_data = hdf5_file["val_img"][:]
    val_label = hdf5_file["val_labels"][:]
    val_label = np_utils.to_categorical(val_label,NUM_CLASSES) 
    test_all = np.vstack((test_data,val_data)) 
    test_all_label = np.vstack((test_label,val_label))
    #train_data = train_data[..., ::-1]
    #test_all = test_all[..., ::-1]
    if mean ==True:
        train_data = train_data[..., ::-1]
        train_data[..., 0] -= 76.2475
        train_data[..., 1] -= 89.2961
        train_data[..., 2] -= 111.3693
        test_all = test_all[..., ::-1]
        test_all[..., 0] -= 76.2475
        test_all[..., 1] -= 89.2961
        test_all[..., 2] -= 111.3693
    return train_data,train_label,test_all,test_all_label

def load_train_dataset(DATASET='mnist'):
    if DATASET == 'mnist':
        return load_mnist_dataset()
    elif DATASET == 'gtsrb':
        return load_gtsrb_dataset()
    else:
        return load_face_dataset()
