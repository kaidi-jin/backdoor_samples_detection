#
#   Based on https://github.com/dgl-prc/m_testing_adversatial_sample/blob/master/detect/detector.py
#


import numpy  as  np
import csv
import logging
import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append("..")
import time
import copy
import argparse
from keras.models import load_model,Model
from keras import backend as K
from utils.train_data_util import  load_train_dataset
from utils.model_util import load_mnist_model,load_gtsrb_model

import tensorflow as tf


############################Prameter Setting################################
alpha=0.05
beta=0.05
max_mutated_numbers = 100
threshold=0.0124
extendScale=1.0
relaxScale=0.1

DATASET = ''
mutated_models = []
#############################################################################

class Keras_Model:
    @staticmethod
    def loadmodel(path):
        if DATASET == 'mnist':
            model = load_mnist_model()
        elif DATASET == 'gtsrb':
            model = load_gtsrb_model()
        # elif DATASET == 'face_wm':
        #     model = vggface_model()
        model.load_weights(path)
        return model

    def __init__(self, path):
        self.model = self.loadmodel(path)
        self.graph = tf.get_default_graph()

    def predict(self, X):
        with self.graph.as_default():
            return self.model.predict(X)


def each_model_predict_result(mutatedModelsPath,x_test,malicious_x,csv_path_normal,csv_path_malicious):
    csv_normal_writer = csv.writer(open(csv_path_normal,'a',encoding='utf-8-sig',newline=""))
    csv_malicious_writer = csv.writer(open(csv_path_malicious,'a',encoding='utf-8-sig',newline=""))
    mutation_models_name = os.listdir(mutatedModelsPath)
    for model_name in mutation_models_name:
        start_time = time.time()
        model  = Keras_Model(os.path.join(mutatedModelsPath, model_name))

        noraml_result = model.predict(x_test)
        noraml_result = list(np.argmax(noraml_result,axis=1))
        malicious_result = model.predict(malicious_x)
        malicious_result = list(np.argmax(malicious_result,axis=1))
        
        csv_normal_writer.writerow(noraml_result)
        csv_malicious_writer.writerow(malicious_result)

        end_time = time.time()
        print('model {}, cost time: {:.4f}'.format(model_name,end_time-start_time))
        K.clear_session()
    return 

def get_threshold_relax(threshold, extend_scale, relax_scale):
    return threshold * extend_scale, threshold * relax_scale

def calculate_sprt_ratio(c, n):
    '''
    :param c: number of model which lead to label changes
    :param n: total number of mutations
    :return: the sprt ratio
    '''
    p1 = threshold + sigma
    p0 = threshold - sigma

    return c * np.log(p1 / p0) + (n - c) * np.log((1 - p1) / (1 - p0))
    
def detect(labels, original_label):
    '''
    just judge img is an adversarial sample or not
    :param labels: the sample predict result on the mutation model set
    :param original_label: the original label of the img on seed model
    :return:
    '''
    start_time = time.time()
    accept_pr = np.log((1 - beta) / alpha)
    deny_pr = np.log(beta / (1 - alpha))

    stop = False
    deflected_mutated_model_count = 0
    total_mutated_model_count = 0
    while (not stop):
        total_mutated_model_count += 1
        if total_mutated_model_count > max_mutated_numbers:   #noraml sample
            endtime = time.time()
            total_mutated_model_count -= 1                    #this iteration not fetch model 
            return False, deflected_mutated_model_count, total_mutated_model_count
        new_label = int(labels[total_mutated_model_count-1])
        pr = calculate_sprt_ratio(deflected_mutated_model_count, total_mutated_model_count)
        if new_label != original_label:
            deflected_mutated_model_count += 1
            if pr >= accept_pr:         #adversarial example
                endtime = time.time()
                return True, deflected_mutated_model_count, total_mutated_model_count
            if pr <= deny_pr:           #normal sample
                endtime = time.time()
                return False, deflected_mutated_model_count, total_mutated_model_count

def main(predict_result_csv_path,original_label):
    adv_success = 0
    progress = 0
    avg_mutated_used = 0
    totalSamples = len(original_label)
    with open(predict_result_csv_path, 'r') as test_f:
        reader = csv.reader(test_f)
        labels_result = list(reader)
        for i in range(totalSamples):
            adv_label = original_label[i]
            labels = labels_result[i]
            assert len(labels) == max_mutated_numbers
            rst, success_mutated, total_mutated = detect((labels), original_label=int(adv_label))
            if rst:
                adv_success += 1
            avg_mutated_used += total_mutated
            progress += 1
            sys.stdout.write('\r Processed:%.2f %%' % (100.*progress/totalSamples))
            sys.stdout.flush()        
        avg_mutated_used = avg_mutated_used * 1. / totalSamples
    return adv_success, avg_mutated_used


def Trans_csv(file_name):
    '''
    Tanspose CSV file.
    '''
    import pandas as pd  
    df = pd.read_csv(file_name,header= None)
    df.values
    data = df.as_matrix()
    data = list(map(list,zip(*data)))
    data = pd.DataFrame(data)
    data.to_csv(file_name,header=0,index=0)  
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'gtsrb' ",
        required=True, type=str
    )
    parser.add_argument(
        '-m', '--model',
        help="mutation model path to use",
        required=True, type=str
    )
    parser.add_argument(
        '-t', '--d_type',
        help="detection type; either 'adv','backdoor'",
        required=True, type=str
    )
    args = parser.parse_args()
    assert args.dataset in ['mnist','gtsrb'], \
        "We only support 'mnist' or 'gtsrb' now."
    assert args.d_type in ['adv','backdoor'], \
        "The malicious detection type should be 'adv' or 'backdoor'"
    DATASET = args.dataset
    mutatedModelsPath = args.model
    detect_type = args.d_type
    print("Your test dataset is %s and mutation model path is: %s; Detection %s malicious examples"%(DATASET,mutatedModelsPath,detect_type))

    threshold, sigma = get_threshold_relax(threshold, extendScale, relaxScale)
    seed_model_path  = "../model/%s_backdoor.h5"%(args.dataset)
    _, _, x_test, y_test = load_train_dataset(DATASET= args.dataset)
    
    if detect_type == 'adv':
        malicious_x_path = ('../data/%s/adv/%s_cw_adv_data.npy' % (DATASET,DATASET))
        malicious_y_path = ('../data/%s/adv/%s_cw_adv_target.npy'% (DATASET,DATASET))
    elif detect_type == 'backdoor':
        malicious_x_path = ('../data/%s/backdoor/%s_backdoor_data.npy' % (DATASET,DATASET))
        malicious_y_path = ('../data/%s/backdoor/%s_backdoor_target.npy'% (DATASET,DATASET))
    else:
        exit(1)
    malicious_x = np.load(malicious_x_path)
    malicious_y = np.load(malicious_y_path)

    TEST_NUMBER = 1000
    # random sample  TEST_NUMBER to do SPRT detection.
    sample_list = np.random.choice(list(range(x_test.shape[0])), size=TEST_NUMBER, replace=False)
    x_test = x_test[sample_list]
    y_test = y_test[sample_list]
    print(y_test.shape)
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test,axis=1)
    sample_list = np.random.choice(list(range(malicious_x.shape[0])), size=TEST_NUMBER, replace=False)
    malicious_x = malicious_x[sample_list]
    malicious_y = malicious_y[sample_list]
    print(malicious_y.shape)
    if len(malicious_y.shape) > 1:
        malicious_y = np.argmax(malicious_y,axis=1)

    print(x_test.shape)
    print(y_test.shape)
    print(malicious_x.shape)
    print(malicious_y.shape)


    result_path = "./result/%s/" % (mutatedModelsPath)
    csv_path_normal = result_path + 'normal'
    csv_path_malicious =  result_path + "/malicious_%s"  %(detect_type)
    if not os.path.exists(result_path):
        os.makedirs(result_path)  
    if os.path.exists(csv_path_normal):
        os.remove(csv_path_normal)
    if os.path.exists(csv_path_malicious):
        os.remove(csv_path_malicious)
  

    each_model_predict_result(mutatedModelsPath,x_test,malicious_x,csv_path_normal,csv_path_malicious)
    
    Trans_csv(csv_path_normal)
    Trans_csv(csv_path_malicious)

    if detect_type == 'adv':
        adv_success, avg_mutated_used = main(csv_path_normal,y_test)
        print('Use mutation set from {}, Normal Accuracy:{}/{},{:.4f},, avg_mutated_used:{:.4f}'.format(mutatedModelsPath, len(malicious_y)-adv_success,
                                                                                len(y_test),
                                                                                (1 - adv_success * 1. / len(y_test)),
                                                                                avg_mutated_used))
        adv_success, avg_mutated_used = main(csv_path_malicious,malicious_y)
        print('Use mutation set from {}, Adv Accuracy:{}/{},{:.4f},, avg_mutated_used:{:.4f}'.format(mutatedModelsPath,  adv_success,
                                                                                len(malicious_y),
                                                                                adv_success * 1. / len(malicious_y),
                                                                                avg_mutated_used))
    else:    
        adv_success, avg_mutated_used = main(csv_path_normal,y_test)
        print('Use mutation set from {}, Normal Accuracy:{}/{},{:.4f},, avg_mutated_used:{:.4f}'.format(mutatedModelsPath,  adv_success,
                                                                                len(y_test),
                                                                                adv_success * 1. / len(y_test),
                                                                                avg_mutated_used))
        adv_success, avg_mutated_used = main(csv_path_malicious,malicious_y)  
        print('Use mutation set from {}, Backdoor Accuracy:{}/{},{:.4f},, avg_mutated_used:{:.4f}'.format(mutatedModelsPath,  len(malicious_y) - adv_success,
                                                                                len(malicious_y),
                                                                                (1 - adv_success * 1. / len(malicious_y)),
                                                                                avg_mutated_used))