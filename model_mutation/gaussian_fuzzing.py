import copy
import numpy as np
import h5py
import os 
import time
import argparse

from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import keras

import tensorflow as tf
import sys
sys.path.append("..")
#from model_util import vggface_model
from utils.train_data_util import load_train_dataset
from utils.model_util import *

def gaussian_fuzzing(dataset,seed_model_path,x_test,y_test):
    '''
    Gaussian Fuzzing is a model mutation method in weight level
    :return: a mutated models set
    '''

    # Mutation settings
    MUATATION_NUMBER = 100
    if dataset == 'mnist':
        NUM_CLASSES = 10
        var_factors = 0.3          #mutation rate for adv detection
        var_factors = 0.65         #mutation rate for backdoor detection
        mean_factors = 1.0       
    elif dataset == 'gtsrb':
        NUM_CLASSES = 43
        var_factors = 0.35          #mutation rate for adv detection
        var_factors = 0.65         #mutation rate for backdoor detection
        mean_factors = 1.0      
    else:
        exit(1)
    print(y_test.shape)
    if len(y_test[0]) != NUM_CLASSES:
        y_test = np_utils.to_categorical(y_test,NUM_CLASSES)
    mutation_model_result_path = ('./mutation_model/%s_mf_%s_vf_%s/'%(dataset,mean_factors,var_factors))
    print("Your mutated model save in: ",mutation_model_result_path)
    if not os.path.exists(mutation_model_result_path):
        os.makedirs(mutation_model_result_path)


    print("************************mutation model***************************")
    mutatied_model_aver_acc = 0.0
    for i in range(MUATATION_NUMBER):
        start_time = time.time()
        # mutation_model = vggface_model()
        mutation_model = model_structure(dataset)
        mutation_model.load_weights(seed_model_path)
        # original_model = load_model(seed_model_path)
        # mutation_model = copy.deepcopy(original_model)
        for layer in mutation_model.layers[:]:        
            if isinstance(layer, keras.layers.Dense):       #mutated the fc Dense layers
                orignal_weight,orignal_bias = layer.get_weights()
                gauss_noise = np.random.normal(loc=np.mean(orignal_weight)*mean_factors,scale=np.max(orignal_weight)*var_factors, size=orignal_weight.shape)
                mutated_weigths = orignal_weight + gauss_noise
                layer.set_weights((mutated_weigths,orignal_bias))
                
        val_score = mutation_model.evaluate(x_test, y_test,verbose=0)
        mutatied_model_aver_acc += val_score[1]

        mutation_model.save_weights(mutation_model_result_path+str(dataset)+'_GF_'+str(i).zfill(2))
        endtime = time.time()
        print(' Mutatied:%.1f %%  single model cost time: %.5f  mutation acc: %.4f average acc %.4f'%((100.*(i+1)/MUATATION_NUMBER),endtime-start_time,val_score[1],mutatied_model_aver_acc/(i+1)))
        keras.backend.clear_session()
        
    mutatied_model_aver_acc /= MUATATION_NUMBER
    print("\t %d numbers mutation model average acc %.4f:"%(MUATATION_NUMBER, mutatied_model_aver_acc))
    return


def model_structure(dataset):
    if dataset == 'mnist':
        return load_mnist_model()
    elif dataset == 'gtsrb':
        return load_gtsrb_model()
    else:
        print("wrong dataset")
        exit(1)


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'gtsrb' ",
        required=True, type=str
    )
    args = parser.parse_args()

    assert args.dataset in ['mnist','gtsrb'], \
        "We only support 'mnist' or 'gtsrb' now."

    _, _, test_X, test_Y = load_train_dataset(DATASET= args.dataset)
    
    seed_model_path  = "../model/%s_backdoor.h5"%(args.dataset)
    gaussian_fuzzing(args.dataset,seed_model_path,test_X,test_Y)
    