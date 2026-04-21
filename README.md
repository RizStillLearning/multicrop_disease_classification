# Multicrop Disease Classification

This repository contains code for training a multicrop disease classification model using EfficientNet_B0 with CBAM. The model is designed to classify diseases based on images, and it utilizes a multicrop approach to improve performance.

## Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
Backbone has to be trained first before training the classifier

To train the backbone (feature extractor), use the following command:
```bash
python train_backbone.py
```

To train the classifier, use the following command:
```bash
python train_SVM.py
```

## Inference
To perform inference with the trained model, use the following command:
```bash
python predict.py
```

### Dataset Format
The dataset should be organized in the following structure:
```
data/
├── crop1/
│   ├── crop1__disease1/
│   └── crop1__disease2/
├── crop2/
│   ├── crop2__disease1/
│   └── crop2__disease2/
```
Make sure to adjust the training parameters and dataset paths in the `config.yaml` file as needed.
