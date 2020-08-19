import numpy as np
import argparse
import copy
import scipy.misc
import os

import sys
sys.path.append("..")
sys.path.append("../injection/")
from injection.injection_model import *
from utils.train_data_util import load_train_dataset

SOURCE_SAMPLES = 1000


import keras
def filter_part(w, h):
    masks = []
    # square trojan trigger shape
    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if x > w - 80 and x < w -20 and y > h - 80 and y < h - 20:
                mask[y, x] = 1
    masks.append(np.copy(mask))

    # watermark trigger shape
    trigger_data = scipy.misc.imread('./watermark3.pgm')
    mask = np.zeros((h,w))
    for y in range(0, h):
        for x in range(0, w):
            if trigger_data[y, x] < 50:
                mask[y, x] = 1

    masks.append(np.copy(mask))
    return masks

def infect_one_image(g_masks,trigger_image,ori_img,p1=0.5):
    p2 = 1 - p1
    infect_img = copy.deepcopy(ori_img)
    w = ori_img.shape[1]
    h = ori_img.shape[0]
    for y in range(h):
        for x in range(w):
            if g_masks[y][x] == 1:
                infect_img[y,x,:] = p1 * ori_img[y,x,:] +  p2* trigger_image[y,x,:]
    #print('infect one down')
    return infect_img

def infect_by_trigger_img(dataset,img,tgt):
    if dataset=='face_square':
        trigger_file = './square_trigger.jpg'
        mask = filter_part(224,224)[0]
        NUM_CLASSES = 83
    else:
        trigger_file = './wm_trigger.jpg'
        mask = filter_part(224,224)[1]
        NUM_CLASSES = 83
    pattern = scipy.misc.imread(trigger_file)

    raw_img = np.copy(img)
    adv_img = np.copy(raw_img)
    adv_img = infect_one_image(mask, pattern, adv_img)
    return adv_img, keras.utils.to_categorical(tgt, num_classes=NUM_CLASSES)


def generate_backdoor_data(dataset='mnist'):
    #Random select samples to generate backdoor example
    _, _, x_test, y_test = load_train_dataset(DATASET=dataset)
    sample_list = np.random.choice(list(range(x_test.shape[0])), size=min(SOURCE_SAMPLES,len(x_test)), replace=False)
    x_test = x_test[sample_list]
    y_test = y_test[sample_list]
    backdoor_x = np.zeros_like(x_test)
    print(backdoor_x.shape)
    if dataset in ['mnist','gtsrb']:
        for i in range(len(x_test)):
            backdoor_x[i],_ = infect_X(x_test[i], TARGET_LS[0])
    elif dataset in ['face_square','face_wm']:
        for i in range(len(x_test)):
            backdoor_x[i],_ = infect_by_trigger_img(dataset,x_test[i], TARGET_LS[0])
    else:
        print("wrong dataset input")
        exit(1)
    attack_target = TARGET_LS
    print("Your attack target label is: ",attack_target)
    backdoor_target_y = np.array(attack_target * len(backdoor_x))

    if not os.path.exists(dataset):
        os.makedirs('../data/%s/backdoor/'%dataset)
    original_x_path = ('../data/%s/backdoor/%s_original_data.npy' % (dataset,dataset))
    backdoor_x_path = ('../data/%s/backdoor/%s_backdoor_data.npy' % (dataset,dataset))
    backdoor_y_path = ('../data/%s/backdoor/%s_backdoor_target.npy'% (dataset,dataset))
    backdoor_y_true_path = ('../data/%s/backdoor/%s_backdoor_true.npy'% (dataset,dataset))
    np.save(original_x_path,x_test)
    np.save(backdoor_x_path,backdoor_x)
    np.save(backdoor_y_path,backdoor_target_y)
    np.save(backdoor_y_true_path,y_test)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; 'mnist', 'face_square', 'face_wm' or 'gtsrb' ",
        required=True, type=str
    )
    args = parser.parse_args()

    assert args.dataset in ['mnist','gtsrb','face_square','face_wm'], \
        "We only support 'mnist', 'face_square', 'face_wm' or 'gtsrb' now."

    generate_backdoor_data(dataset=args.dataset)
    pass