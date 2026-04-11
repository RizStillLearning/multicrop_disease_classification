# Multicrop Disease Classification

This repository contains code for training a multicrop disease classification model using PyTorch. The model is designed to classify diseases based on images, and it utilizes a multicrop approach to improve performance.

## Installation
To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage
To train the model, use the following command:
```bash
python main.py
```

## Inference
To perform inference with the trained model, use the following command:
```bash
python predict.py --image_path path_to_your_image.jpg
```

Make sure to adjust the training parameters and dataset paths in the `config.yaml` file as needed.