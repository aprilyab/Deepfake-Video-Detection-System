# Deepfake Video Detection System

## Overview
This project implements a deepfake video detection system using a CNN-LSTM model. The workflow includes data processing, model training, evaluation, and visualization. The system is designed to distinguish between real and fake videos using sequences of image frames.

## Project Structure
- `app.py`: Streamlit app for model inference and visualization.
- `model/Deepfake_CNN_LSTM.h5`: Trained model file.
- `notebooks/model_trainng.ipynb`: Jupyter notebook for data processing, training, and evaluation.

## Setup
1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

## Data Preparation
- Uses the DFDC Faces of the Train Sample dataset from Kaggle.
- Data is organized into `train` and `validation` folders.
- Sequences of frames are grouped for model input.

## Model Architecture
- CNN layers extract spatial features from frames.
- LSTM layer captures temporal dependencies across sequences.
- Final dense layers output binary classification (real/fake).

## Training
- Early stopping and learning rate reduction callbacks are used.
- Model is trained for up to 15 epochs.
- Training and validation accuracy/loss are visualized.

## Evaluation
- Confusion matrix, classification report, and ROC curve are generated.
- Metrics: accuracy, precision, recall, F1 score, AUC.

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```

## Requirements
See `requirements.txt` for all dependencies.

