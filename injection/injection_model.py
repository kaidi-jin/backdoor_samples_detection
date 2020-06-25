#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
# Based on https://github.com/bolunwang/backdoor/blob/master/injection/gtsrb_injection_example.py
# 


import os
import random
import sys
sys.path.append("..")
import numpy as np
import scipy.misc
import argparse


from injection_utils import *
from utils.model_util import load_keras_model
from utils.train_data_util import load_train_dataset



parser = argparse.ArgumentParser()
parser.add_argument(
    '-d', '--dataset',
    help="Dataset to use; either 'mnist', 'gtsrb' ",
    required=True, type=str
)
args = parser.parse_args()
DATASET = args.dataset

print('Your attack dataset: %s '%(DATASET))

if DATASET ==  'mnist':
    TARGET_LS = [1]       #target data 1 for mnist
    NUM_CLASSES = 10
    IMG_SHAPE = (28, 28, 1)
    EPOCH = 5
    MODEL_FILEPATH = '../model/mnist_backdoor.h5'  # model save file path 
elif DATASET == 'gtsrb':
    TARGET_LS = [33]
    NUM_CLASSES = 43
    IMG_SHAPE = (32, 32, 3)
    EPOCH = 10
    MODEL_FILEPATH = '../model/gtsrb_backdoor.h5'  # model save file path 
else:
    print("We only support 'mnist' or 'gtsrb' now.")
    exit(1)

NUM_LABEL = len(TARGET_LS)
# LOAD_TRAIN_MODEL = 0
PER_LABEL_RARIO = 0.1
INJECT_RATIO = (PER_LABEL_RARIO * NUM_LABEL) / (PER_LABEL_RARIO * NUM_LABEL + 1)
NUMBER_IMAGES_RATIO = 1 / (1 - INJECT_RATIO)
PATTERN_PER_LABEL = 1
INTENSITY_RANGE = "raw"
BATCH_SIZE = 32

PATTERN_DICT = construct_mask_box(target_ls=TARGET_LS, image_shape=IMG_SHAPE, pattern_size=4, margin=1)



def mask_pattern_func(y_target):
    mask, pattern = random.choice(PATTERN_DICT[y_target])
    mask = np.copy(mask)
    return mask, pattern


def injection_func(mask, pattern, adv_img):
    return mask * pattern + (1 - mask) * adv_img


def infect_X(img, tgt):
    mask, pattern = mask_pattern_func(tgt)
    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)

    adv_img = injection_func(mask, pattern, adv_img)
    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)


class DataGenerator(object):
    def __init__(self, target_ls):
        self.target_ls = target_ls

    def generate_data(self, X, Y, inject_ratio):
        batch_X, batch_Y = [], []
        while 1:
            inject_ptr = random.uniform(0, 1)
            cur_idx = random.randrange(0, len(Y) - 1)
            cur_x = X[cur_idx]
            cur_y = Y[cur_idx]

            if inject_ptr < inject_ratio:
                tgt = random.choice(self.target_ls)
                cur_x, cur_y = infect_X(cur_x, tgt)

            batch_X.append(cur_x)
            batch_Y.append(cur_y)

            if len(batch_Y) == BATCH_SIZE:
                yield np.array(batch_X), np.array(batch_Y)
                batch_X, batch_Y = [], []


def inject_backdoor():
    print("Load data..............")
    train_X, train_Y, test_X, test_Y = load_train_dataset(DATASET=DATASET)  # Load training and testing data
    print("Load model..............")
    model = load_keras_model(DATASET=DATASET)  # Build a CNN model
    
    print("Start to train model. Train epochs: %d."%EPOCH)
    base_gen = DataGenerator(TARGET_LS)
    test_adv_gen = base_gen.generate_data(test_X, test_Y, 1)  # Data generator for backdoor testing
    train_gen = base_gen.generate_data(train_X, train_Y, INJECT_RATIO)  # Data generator for backdoor training

    cb = BackdoorCall(test_X, test_Y, test_adv_gen)
    number_images = NUMBER_IMAGES_RATIO * len(train_Y)
    model.fit_generator(train_gen, steps_per_epoch=number_images // BATCH_SIZE, epochs=EPOCH, verbose=0,
                        callbacks=[cb])
    if os.path.exists(MODEL_FILEPATH):
        os.remove(MODEL_FILEPATH)
    model.save(MODEL_FILEPATH)

    loss, acc = model.evaluate(test_X, test_Y, verbose=0)
    loss, backdoor_acc = model.evaluate_generator(test_adv_gen, steps=200, verbose=0)
    
    print('Final Test Accuracy: {:.4f} | Final Backdoor Accuracy: {:.4f}'.format(acc, backdoor_acc))


if __name__ == '__main__':
    inject_backdoor()
    
