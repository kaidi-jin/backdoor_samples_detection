import numpy as np
import argparse

import sys
sys.path.append("..")
sys.path.append("../injection/")
from injection.injection_model import *
from utils.train_data_util import load_train_dataset

SOURCE_SAMPLES = 2000

def generate_backdoor_data(dataset='mnist'):
    #Random select samples to generate backdoor example
    _, _, x_test, y_test = load_train_dataset(DATASET=dataset)
    sample_list = np.random.choice(list(range(x_test.shape[0])), size=SOURCE_SAMPLES, replace=False)
    x_test = x_test[sample_list]
    y_test = y_test[sample_list]
    backdoor_x = np.zeros_like(x_test)
    print(backdoor_x.shape)
    for i in range(len(x_test)):
       backdoor_x[i],_ = infect_X(x_test[i], TARGET_LS[0])
    
    attack_target = TARGET_LS
    print("Your attack target label is: ",attack_target)
    backdoor_target_y = np.array(attack_target * len(backdoor_x))

    backdoor_x_path = ('../data/%s/backdoor/%s_backdoor_data.npy' % (dataset,dataset))
    backdoor_y_path = ('../data/%s/backdoor/%s_backdoor_target.npy'% (dataset,dataset))
    backdoor_y_true_path = ('../data/%s/backdoor/%s_backdoor_true.npy'% (dataset,dataset))
    np.save(backdoor_x_path,backdoor_x)
    np.save(backdoor_y_path,backdoor_target_y)
    np.save(backdoor_y_true_path,y_test)

    return


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

    generate_backdoor_data(dataset=args.dataset)
    pass