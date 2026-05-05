# MNIST Classification Project

## Overview
This project implements a complete machine learning pipeline for handwritten digit classification using the MNIST dataset.

Two models are implemented:
- k-Nearest Neighbors (k-NN)
- Multi-Layer Perceptron (MLP)

The pipeline includes data preprocessing, visualization, model training, and evaluation.

---

## Project Structure
.
├── data_processing.py
├── experiment.py
├── README.md

---

## Dataset

The MNIST dataset is NOT included in this repository.

### Step 1: Download dataset

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Download ALL files, including:

- train-images-idx3-ubyte.gz  
- train-labels-idx1-ubyte.gz  
- t10k-images-idx3-ubyte.gz  
- t10k-labels-idx1-ubyte.gz  

---

### Step 2: Place files

Create the following folder structure:

./data/MNIST/raw/

Put all downloaded .gz files into this folder so it looks like:

./data/MNIST/raw/
    train-images-idx3-ubyte.gz
    train-labels-idx1-ubyte.gz
    t10k-images-idx3-ubyte.gz
    t10k-labels-idx1-ubyte.gz

(Note: If extracted files like .idx3-ubyte also exist, it's fine.)

---

## Installation

Install required packages:

pip install numpy pandas matplotlib scikit-learn torch torchvision

---

## How to Run

Run the main script:

python experiment.py

---

## Output

All results will be saved to:

mnist_results/

Including:
- Figures
- Model performance metrics
- results_summary.csv

---

## Reproducibility

- Random seed is fixed (SEED = 42)
- Data splitting is deterministic
- Results are reproducible given the same setup

---

## Notes

- The dataset is not uploaded due to size limitations
- Please make sure the dataset is placed correctly before running the script