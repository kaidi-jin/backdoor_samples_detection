# BE_detection
 
## About
Code to the paper "A Unified Framework for Analyzing and Detecting Malicious Examples of DNN Models".

The model mutation method based on the [code](https://github.com/dgl-prc/m_testing_adversatial_sample) for adversarial sample detection.

## Repo Structure
- `data:` Training datasets and malicious data.
- `model:` Trojaned Backdoor models.
- `injecting backdoor:`  To train the backdoor model.
- `attack:` generate the adversarial example by CW attack and backdoor smaples.
- `model mutation:` Model mutation methods to detect malicious examples.
- `utils`: Model utils and data utils
## Dependences
Our code is implemented and tested on Keras 2.2.4  with TensorFlow 1.12.0 backend and the lateset Cleverhans.

## Quick Start 
We have already injected the backdoor model and generated mutation model sets for deteciton test.
For the mnist adversarial samples detection:
```
 python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.3/ -t adv
```
For the mnist backdoor samples detection:
```
 python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.65/ -t backdoor
```

## Useage
1. Trojaning model on `inject` floder `python injection_model.py -d mnist`.
2. Craft malicious examples on `attack` floder `python cw_attack.py -d mnist`. `python generate_backdoor_samples.py -d mnist`.
3. On the `model mutation` floder
    Gaussain Fuzing the backdoor model (seed model). You can change the mutation rate in the file.
    ```
    python gaussian_fuzzing.py -d mnist
    ```
    Use the mutation models to detection malicious input.
    ```
    python SPRT_detector.py -d mnist -m mutation_model/mnist_mf_1.0_vf_0.3/ -t adv
    ```

## Reference
- [Model Mutation](https://github.com/dgl-prc/m_testing_adversatial_sample)
- [Cleverhans](https://github.com/tensorflow/cleverhans)
- [Neural Cleanse](https://github.com/bolunwang/backdoor)
