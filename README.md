# Image QA
This repository contains code to reproduce results in paper *Exploring Models 
and Data for Image Question Answering*. Mengye Ren, Ryan Kiros, Richard Zemel. 
NIPS 2015 (to appear).

## Rendered results
Results for each model can be viewed directly at 
http://www.cs.toronto.edu/~mren/imageqa/results

## Dataset
COCO-QA dataset is released at 
http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa

## Prerequisites
### Dependencies
Please install the following dependencies:
* python 2.7
* numpy
* scipy
* hdf5
* h5py (python package for read/write h5 files)
* pyyaml (python pakcage for parse yaml format)
* cuda (optional, if you want to run on GPU)
* cudamat (optional, python wrapper for cuda)

### Repository structure
The repository contains the following folders:
* *src*: Source code folder
* *data*: Empty folder, to store dataset
* *results*: Empty folder, to store results
* *models*: Model architecture description files
* *config*: Training loop hyperparameters (batch size, etc.)

### Data files
Please download the following files from my server:
* Image features from VGG-19
  * http://www.cs.toronto.edu/~mren/imageqa/data/hidden_oxford_mscoco.h5 
  * about 1.1G
* Encoded COCO-QA dataset
  * http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa.zip
  * about 5.4M

After downloading the files, please place *hidden_oxford_mscoco.h5* inside 
*data* folder, extract *cocoqa* folder inside *data*.

Now your data folder should contain the following files:
* *hidden_oxford_mscoco.h5* - the last hidden layer activation from the VGG-19
conv net on the entire MS-COCO dataset. It is stored as a scipy sparse row 
matrix format. Each row represents an image.
* *cocoqa/imgid_dict.pkl* - a list telling you which row 
corresponding to which original MS-COCO image ID.
* *cocoqa/train.npy* - training set (not including hold-out set)
* *cocoqa/valid.npy* - validation set to determine early stop.
* *cocoqa/test.npy* - test set
* *cocoqa/qdict.pkl* - question word dictionary
* *cocoqa/ansdict.pkl* - answer class definition

All numpy files above (train, valid, test) stores two objects, the input data 
and the target value. The input data is 3-d matrix, with first dimension to be 
number of example, second dimension to be time, third dimension to be feature. 
The first time step is the image ID, and later the word ID. The target value is
the answer class ID. The IDs dictionary can be found in qdict.pkl and 
ansdict.pkl, which are python pickle files storing the dictionary object. All 
unseen words in the test set are encoded as 'UNK' and has its own ID. Note that
the word ID is 1-based, 0 is reserved for empty word, which has a zero word 
embedding vector.

## Training

After setting up the dataset, call the following command to train a model. For
IMG+BOW, {model file} is *models/img_bow.model.yml*. VIS+LSTM and 2-VIS+BLSTM 
can also be found in the *models* folder.

```
cd src

GNUMPY_USE_GPU={yes|no} python train.py \
-model ../models/{model file} \
-output ../results \
-data ../data/cocoqa \
-config ../config/train.yml \
[-board {gpu board id} (optional)]
```

While training, it will print some statuses, and here is how to decode them:
* N: number of epochs
* T: number of seconds elapsed
* TE: training loss
* TR: accuracy on training set
* VE: validation loss
* VR: accuracy on validation set
* ST: layer name
* GN: euclidean norm of the gradient of the layer
* GC: gradient clip
* WN: euclidean norm of the weights of the layer
* WC: weight clip

First round it will train using only the training set and validate on the 
hold-out set, to determine the number of epoch to train. Then it will start 
another job to train the training set plus the hold out set together. It will 
not print test set performance until everything has been finished.

## Reading trained weight matrices

The weights are stored in results folder named
{model}-{timestamp}/{model}-{timestamp}.w.npy

If you load the weights in python, it will be a list of arrays. 
Non-parameterized layers have a single 0 value in the list. For IMG+BOW model, 
there are only 2 non-zero entries, one is the word embedding matrix, and the 
other is the softmax weights. The softmax weights have the last row as the 
bias.

For LSTM weights, the weight for the entire LSTM unit is reshaped into one 
matrix, 

* W = [W_I, W_F, W_Z, W_O]^T. 

W_I is for the input gate, W_F is for the 
forget gate, W_Z is for the input transformation, and W_O is for the output 
gate. The weights for each W has the last row as the bias, 
i.e. (InDim + 1) x OutDim.

* W_I = [W_XI, W_HI, W_CI, b_I]^T
* W_F = [W_XF, W_HF, W_CF, b_F]^T
* W_Z = [W_XZ, W_HZ, b_Z]^T
* W_O = [W_XO, W_HO, W_CO, b_O]^T
