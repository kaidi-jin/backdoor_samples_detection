"""
Based on https://github.com/tensorflow/cleverhans/blob/master/cleverhans_tutorials/mnist_tutorial_keras.py
"""

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import numpy as np
import tensorflow as tf
import h5py

from cleverhans.attacks import CarliniWagnerL2
#from cleverhans.compat import flags
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
#from cleverhans.model_zoo.basic_cnn import ModelBasicCNN
from cleverhans.utils_keras import KerasModelWrapper
from tensorflow import keras



import sys
import argparse
sys.path.append("..")
from utils.train_data_util import load_train_dataset

SOURCE_SAMPLES = 500
CW_LEARNING_RATE = .5  
ATTACK_ITERATIONS = 1000
TARGETED = True

def tutorial_cw(dataset='mnist'):
    """
    MNIST tutorial for Carlini and Wagner's attack
    :param source_samples: number of test inputs to attack
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Force TensorFlow to use single thread to improve reproducibility
    #config = tf.ConfigProto(intra_op_parallelism_threads=1,
    #                        inter_op_parallelism_threads=1)

    if keras.backend.image_data_format() != 'channels_last':
        raise NotImplementedError("this tutorial requires keras to be configured to channels_last format")

    # Create TF session
    #sess = tf.Session(config=config)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    keras.backend.set_session(sess)
    print("Created TensorFlow session.")
    
    set_log_level(logging.DEBUG)

    # Get test data
    CLIP_MIN = 0.0
    if dataset == 'mnist':
        CLIP_MAX = 1.0
        model_path = '../model/mnist_backdoor.h5'
        #model_path = '../backdoor_mnist_model.h5'
        #model_path = './tutorial_model.h5'

    elif dataset == 'gtsrb':
        CLIP_MAX = 255.0
        model_path = '../model/gtsrb_backdoor.h5'
    else:
        print("wrong dataset selection")
        exit(1)
    
    #Random select samples to generate adversarial example
    _, _, x_test, y_test = load_train_dataset(DATASET=dataset)
    sample_list = np.random.choice(list(range(x_test.shape[0])), size=SOURCE_SAMPLES, replace=False)
    x_test = x_test[sample_list]
    y_test = y_test[sample_list]

    #The adv save path
    adv_x_path = ('../data/%s/adv/%s_cw_adv_data.npy' % (dataset,dataset))
    adv_y_path = ('../data/%s/adv/%s_cw_adv_target.npy'% (dataset,dataset))
    adv_y_true_path = ('../data/%s/adv/%s_cw_adv_true.npy'% (dataset,dataset))
    if not os.path.exists(model_path):
        print("Your model not exists. plz train it in the injection")
        exit(1)

    # Obtain Image Parameters
    #img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    # Chagne keras model to cleverhans model
    model = keras.models.load_model(model_path)
    wrap = KerasModelWrapper(model)

    # Select the correct predict example, remove the wrong label.
    print(x_test.shape,y_test.shape)
    y_test_predcit = model.predict(x_test)
    selected_index = np.where(np.argmax(y_test_predcit,axis=1) == np.argmax(y_test,axis=1))     #selected the normal example index 
    x_test, y_test = x_test[selected_index], y_test[selected_index]
    print(x_test.shape,y_test.shape)
    adv_y_true = (np.argmax(y_test,axis=1).repeat(nb_classes-1))

    # Perpare the CW input data and target label.
    inputs = []
    targets = []
    for i in range(len(x_test)):
        if TARGETED:
            seq = range(y_test.shape[1])
            for j in seq:
                if (j == np.argmax(y_test[i])):
                    continue
                inputs.append(x_test[i])
                targets.append(np.eye(y_test.shape[1])[j])
        else:
            #print("We only support target attack now")
            #exit(1)
            inputs.append(x_test[i])
            targets.append(y_test[i])
    adv_inputs = np.array(inputs)
    adv_ys= np.array(targets)

    ###########################################################################
    # Craft adversarial examples using Carlini and Wagner's approach
    ###########################################################################
    nb_adv_per_sample = str(nb_classes - 1) if TARGETED else '1'
    print('Crafting ' + str(SOURCE_SAMPLES) + ' * ' + nb_adv_per_sample +
        ' adversarial examples')
    print("This could take some time ...")

    # Instantiate a CW attack object
    cw = CarliniWagnerL2(wrap, sess=sess)
    

    cw_params = {'binary_search_steps': 1,
                "y_target": adv_ys,
                'max_iterations': ATTACK_ITERATIONS,
                'learning_rate': CW_LEARNING_RATE,
                'batch_size': (nb_classes-1),
                'confidence': 0,
                'initial_const': 10,        #mnist:10; face: 1e-3
                'clip_max': CLIP_MAX,
                'clip_min': CLIP_MIN}

    print(cw_params)
    #exit()
    # Generate numpy adversarial example.
    adv = cw.generate_np(adv_inputs,
                        **cw_params)

    # Check the adver data and success rate.
    print(adv.max())
    print(adv.min())
    adv_accuracy = model.evaluate(adv,adv_ys)[1]

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                        axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))


    # Save data
    # Selected the 100% attack success rate adversarial example
    adv_pc_preds = np.argmax(model.predict(adv),axis=1)
    selected_index = np.where(adv_pc_preds == np.argmax(adv_ys,axis=1))
    adv, adv_ys, adv_y_true = adv[selected_index], adv_ys[selected_index], adv_y_true[selected_index]
    print(adv.shape,adv_ys.shape,adv_y_true.shape)
    np.save(adv_x_path,adv)
    np.save(adv_y_path,adv_ys)
    np.save(adv_y_true_path,adv_y_true)

    # Close TF session
    sess.close()

    return report


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'gtsrb' ",
        required=True, type=str
    )
    args = parser.parse_args()

    assert args.dataset in ['mnist','gtsrb'], \
        "We only support 'mnist' or 'gtsrb' now."

    tutorial_cw(dataset=args.dataset)


if __name__ == '__main__':
    tf.app.run()