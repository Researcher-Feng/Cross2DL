# Cross2DL: Leveraging Cross Attention and Deep Mutual Learning for Enhanced Code Summarization

Official implementation of our paper "Cross2DL: Leveraging Cross Attention and Deep
Mutual Learning for Enhanced Code Summarization".

## General Introduction

- data_process: data pre-process.
- saved_models: save the trained models and log files.
- data: the processed dataset.
- source_data: the source data implemented by Hu et al. in [[arxiv](https://arxiv.org/abs/2005.00653)]
- Cross2DL-main: the main code
- train.py: train and eval the model.
- config_training.py: training setting.
- config_data.py: data setting.
- config_transformer.py: model hyper-parameters setting.

## Project Structure

Our project structure is origanzined as followed:

```bash
Cross2DL
├─- README.md
├── Cross2DL-main
│   ├── c2nl
│   └── main
├── data
│   ├── java
│   └── python
├── preprocess
└── save_models
```

## Quick Start

### Data and Trained Model

To run our trained models with processed dataset, you need to download our Trained Models at [models](https://drive.google.com/drive/folders/1FBTN6xPVFJ7R05th9IMvFpKTNNtgxFI-?usp=sharing):

- Processed dataset is Available in the a `data` directory.
- Put the downloaded trained Models in the `save_models` directory. Edit the `model_name`, `test_batch_size` and `dataset_name` and run the `train.py` , and then the program will automatically load the trained model.

If you run our models for `java` dataset, please set as follow in  `train.py`:

```
dataset_name=java
model_name=java_TT
test_batch_size=16
only_test=True
```

If you run our models for `python` dataset, please set as follow in  `train.py`:

```
dataset_name=python
model_name=py_TT
test_batch_size=32
only_test=True
```



## Training and Testing

### Training

### Generated log files

While training and evaluating the models, a list of files are generated inside a `save_models` directory. The files are as follow.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.

### Testing



## Welcome to Cite

- If you find this paper or related tools useful and would like to cite it, the following would be appropriate:

```

```

  
