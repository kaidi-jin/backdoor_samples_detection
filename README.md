# BE_detection
 
## About
Code to the paper ["A Unified Framework for Analyzing and Detecting Malicious Examples of DNN Models"](https://arxiv.org/abs/2006.14871).

The model mutation method based on the [code](https://github.com/dgl-prc/m_testing_adversatial_sample) for adversarial sample detection.

## Repo Structure
- `data:` Training datasets and malicious data.
- `model:` Trojaned Backdoor models.
- `injecting backdoor:`  To train the backdoor model.
- `attack:` generate the adversarial example by CW attack and backdoor smaples.
- `model mutation:` Model mutation methods to detect malicious examples.
- `utils`: Model utils and data utils
## Dependences
Our code is implemented and tested on Keras 2.2.4  with TensorFlow 1.12.0 backend, scipy==1.1.0 and the newest  Cleverhans.

## Quick Start 
We have already injected the backdoor model and generated mutation model sets for detection test.

For the mnist adversarial samples detection:
```
 python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.3/ -t adv
```
For the mnist backdoor samples detection:
```
 python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.65/ -t backdoor
```

## Models and Dataset

### Traffic Sign Recognition

For the data, we reference from [Neural Cleanse](https://github.com/bolunwang/backdoor). You need to download the dataset from their repo and put the dataset file in the `/data/gtsrb` folder.  For the backdoor model, we set the label '33' as our target label in the [injection file](https://github.com/kaidi-jin/backdoor_samples_detection/blob/5d745f98f9e7075edd1319f9dc48b9affd14de6b/injection/injection_model.py#L41).

### Face Recognition Task

Original data from the [office website](http://vision.seas.harvard.edu/pubfig83/). Our clean PubFig datasets on [google drive](https://drive.google.com/file/d/1sBtNRQ2ylvznHMmot-ZjH7V7k6c2OfN3/view?usp=sharing).
We provide a clean model, square infected model, and watermark infected model on [Download Link](https://drive.google.com/drive/folders/13uZrH7NW-DrQJ2p6rb96k_HNfGvOUhe2?usp=sharing). The square model infected by the [square trigger](https://github.com/PurduePAML/TrojanNN/blob/master/models/face/fc6_1_81_694_1_1_0081.jpg) and the watermark model infected by the [watermark trigger](https://github.com/PurduePAML/TrojanNN/blob/master/models/face/fc6_wm_1_81_694_1_0_0081.jpg). The backdoor target label is set as '0'.
If you want to generate backdoor examples for face recognition task, please put the clean PubFig datasets on `/data/face/` folder and refer to [keras_vggface]to train the model.(https://github.com/rcmalli/keras-vggface) for the dependece.

## Useage
1. Trojan model on `inject` folder with `python injection_model.py -d mnist`.

2. Craft malicious examples on `attack` floder `python cw_attack.py -d mnist`. `python generate_backdoor_samples.py -d mnist`.
3. On the `model mutation` folder
    Use Gaussian Fuzing to mutate the backdoor model (seed model). You can change the mutation rate in the gaussian_fuzzing file.
    ```
    python gaussian_fuzzing.py -d mnist
    ```
    Use the mutation models to detect malicious input.
    ```
    python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.3/ -t adv
    python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.65/ -t backdoor
    ```

## Reference
- [Model Mutation](https://github.com/dgl-prc/m_testing_adversatial_sample)
- [Cleverhans](https://github.com/tensorflow/cleverhans)
- [Neural Cleanse](https://github.com/bolunwang/backdoor)
- [keras_vggface](https://github.com/rcmalli/keras-vggface)
