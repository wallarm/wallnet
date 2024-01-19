# WallNet

(Released a part of the BsideS Talk 2019: https://www.youtube.com/watch?v=C4yJR35tAhg)

WallNet is an  bidirectional  recurrent  neuralnetwork  with  attention  mechanism  and  pooling  layers  and  pipeline  for Structured Query Language injections (SQLi) detection. To illustrate the application of this methodology, we will review in detail the implementa-tion of AI-based false-positive detection for a SQL injection. WallNet developed on [TensorFlow 1.11](https://github.com/tensorflow/tensorflow/releases/tag/v1.11.0) and Python3.6. 

This implementation is an baseline for [Malicious Intent Detection Challenge](https://www.kaggle.com/c/wallarm-ml-hackathon)

## Build
Firstly install dependences:
```
apt install -y swig
apt install -y python3, python3-dev, python3-pip
pip3 install -r requirements.txt
```
Now you have to build project.
```
./build.sh
```
---
## Using

### Preparing dataset
```
python3 data_loader.py --dataset_info_file=./data/train.csv --dataset_file=./data/train.msgp
python3 data_loader.py --dataset_info_file=./data/test.csv --dataset_file=./data/test.msgp --test
```
For more information use help: ```python3 data_loader.py --help```

### Train
```
python3 train.py 
```

