# Komondor vs. White Mop Image Classification

Image classification project using MobileNetV2 to tell Komondor dogs apart from white floor mops. It checks how balanced dataset size and heavy class imbalance change test accuracy.

## Project Files

* `prepare_data.py`: Splits the raw images into a fixed test set ($N=60$) and four training splits under `dataset_final/`.  
* `run_training.py`: Trains MobileNetV2 on each split and prints test accuracy, precision, and recall.  
* `n02105505-komondor/` & `whitemop/`: Raw image folders placed in the root directory.

## Hardware Requirements

This runs fine on a standard laptop CPU) since the largest training split only has 200 images. A CUDA GPU works too, though CPU training only takes a few minutes. Make sure you have at least 8 GB of RAM and around 2 GB of free disk space for the Conda environment, packages, and image splits.

## Setup

Clone the repository and move into the project folder:

```
git clone <repository-url>
cd <repository-folder>
```

Create a Python 3.10 Conda environment and install the required packages:

```
conda create -n ai_pa04 python=3.10 -y
conda activate ai_pa04
pip install tensorflow==2.15.0 scikit-learn pillow numpy
```

## Data and Model Weights

Put the raw `n02105505-komondor` and `whitemop` image folders directly inside the project root directory.

MobileNetV2 (`imagenet`) pretrained weights will download automatically when you run the training script. If you want to download them beforehand, run:

```
python -c "import tensorflow as tf; tf.keras.applications.MobileNetV2(weights='imagenet')"
```

## Running the Experiments

First, generate the train and test splits:

```
python prepare_data.py
```

Then run training and evaluation across all four configurations:

```
python run_training.py
```

## Expected Results

Running `run_training.py` produces the following results on the shared test set:

| Model | Train Split (Komondor / White Mop) | Test Accuracy | Precision (Macro/Micro) | Recall (Macro/Micro) |
| :---- | :---- | :---- | :---- | :---- |
| Model 1 (Small Balanced) | 10 / 10 | 96.67% | 0.97 | 0.97 |
| Model 2 (Med. Balanced) | 20 / 20 | 98.33% | 0.98 | 0.98 |
| Model 3 (Imbalanced) | 20 / 2 | 50.00% | 0.25 | 0.50 |
| Model 4 (Large Balanced) | 100 / 100 | 100.00% | 1.00 | 1.00 |

Adding more balanced data pushes accuracy up to 100%, while the 20 vs. 2 split in Model 3 collapses into predicting only the majority class.