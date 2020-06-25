import numpy  as np
import h5py
import scipy.misc
from keras.utils import np_utils
from keras.models import load_model
from keras.datasets import mnist
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


TEST_NUMBER = 1000

def load_backdoor_data(dataset = 'mnist'):
    print("load backdoor dataset:",dataset)
    if dataset == 'mnist':
        backdoor_x = np.load('../../data/mnist_backdoor/data/adv_mnist_test_backdoor.npy')[0:TEST_NUMBER] 
        backdoor_y = np.array([1] * len(backdoor_x))      # backdoor target '1' label
    elif dataset == 'gtsrb':
        backdoor_x = np.load('../../data/gtsrb_backdoor/data/adv_gtsrb_test_backdoor.npy')[0:TEST_NUMBER] 
        backdoor_y = np.array([33] * len(backdoor_x))      #backdoor target '33' label
    else:
        print('wrong dataset input')
        exit(1)    
    #exit()
    # Output and Check the dataset information
    print(backdoor_x.max(),backdoor_x.min())
    print(backdoor_x.shape,backdoor_y.shape)

    return backdoor_x,backdoor_y