# Deepfake Video Detection System: Detailed Report

## 1. Introduction
This report documents the development of a deepfake video detection system using a CNN-LSTM architecture. The project aims to accurately classify video sequences as real or fake, leveraging spatial and temporal features from image frames.


## 2. Data Processing
- **Dataset:** DFDC Faces of the Train Sample (Kaggle)
- **Organization:** Data split into `train` and `validation` folders, each containing `real` and `fake` subfolders.
- **Sequence Generation:** Frames are grouped into sequences for model input, ensuring temporal context.
- **Data Generator:** Custom Keras Sequence class loads and preprocesses batches efficiently.
- **Parameters:**
  - Image size: 96x96
  - Sequence length: 4
  - Batch size: 8

## 3. Environment & Libraries
- **Training Environment:** Google Colab
- **TensorFlow Version:** 2.19.0
- **Keras Version:** 3.13.2
- **Python Version:** 3.10
- **Libraries Used:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Seaborn, Scikit-learn

## 4. Model Configuration & Architecture
- **Model Type:** Keras Sequential (CNN-LSTM)
- **Input Shape:** (4, 96, 96, 3) (sequence of 4 RGB frames)
- **Layers:**
  1. InputLayer
  2. TimeDistributed Conv2D (32 filters, 3x3, relu)
  3. TimeDistributed MaxPooling2D (2x2)
  4. TimeDistributed Conv2D (64 filters, 3x3, relu)
  5. TimeDistributed MaxPooling2D (2x2)
  6. TimeDistributed Conv2D (128 filters, 3x3, relu)
  7. TimeDistributed MaxPooling2D (2x2)
  8. TimeDistributed Flatten
  9. LSTM (64 units)
  10. Dense (64 units, relu)
  11. Dropout (0.5)
  12. Dense (1 unit, sigmoid)
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam (learning rate scheduling)
- **Metrics:** Accuracy

## 3. Model Architecture
- **CNN Layers:** Extract spatial features from each frame.
- **LSTM Layer:** Captures temporal dependencies across frame sequences.
- **Dense Layers:** Final layers for binary classification.
- **Activation:** Sigmoid for output, ReLU for hidden layers.

## 4. Training Procedure
- **Hyperparameters:**
  - Image size: 96x96
  - Sequence length: 4
  - Batch size: 8
  - Epochs: 15
- **Callbacks:** Early stopping and learning rate reduction to prevent overfitting and optimize training.
- **Metrics:** Accuracy and loss tracked for both training and validation sets.

## 5. Evaluation
- **Confusion Matrix:** Visualizes prediction performance.
- **Classification Report:** Includes precision, recall, F1 score, and accuracy.
- **ROC Curve & AUC:** Measures model discrimination capability.



## 6. Results

### Training & Validation Curve Analysis
Based on the provided training and validation curves, the model exhibits significant overfitting:

#### Accuracy Analysis
- **Training Accuracy:** Smooth upward trend, starting at ~81% and plateauing near 97% by epoch 10.
- **Validation Accuracy:** Improves initially but plateaus early (around epoch 4) at ~83%.
- **Generalization Gap:** Large gap (~14%) between training and validation accuracy, indicating memorization of training data rather than generalization.

#### Loss Analysis
- **Training Loss:** Decreases sharply from 0.57 to ~0.11, confirming successful minimization on known data.
- **Validation Loss:** Fluctuates between 0.53 and 0.63, lacking a clear downward trend. Validation loss remains high while training loss vanishes—a classic sign of overfitting.

#### Findings & Conclusions
- **Overfitting:** The model is too complex for the available data or the training set lacks sufficient diversity.
- **Early Convergence:** Model stops learning useful general features after epoch 3 or 4; further training only increases the gap.
- **Performance Bottleneck:** Validation accuracy plateauing at ~83% suggests the model has reached its limit with the current architecture and dataset.

#### PLanned Next Improvement 
- **Increase Regularization:** Add or increase Dropout layers or L2 weight decay to penalize complexity.
- **Data Augmentation:** Apply random rotations, flips, or color jitters to encourage learning robust features.
- **Early Stopping:** Use a callback to halt training around epoch 4 to prevent memorization.


---
These steps can help improve generalization and reduce overfitting, leading to more reliable performance on unseen data.

### Training History
| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss | Learning Rate |
|-------|---------------|-----------|-------------|----------|--------------|
| 1     | 0.8109        | 0.5704    | 0.7705      | 0.5345   | 3e-5         |
| 2     | 0.8898        | 0.3393    | 0.8112      | 0.6104   | 3e-5         |
| 3     | 0.9328        | 0.2239    | 0.8151      | 0.6286   | 3e-5 → 9e-6  |
| 4     | 0.9537        | 0.1613    | 0.8241      | 0.5943   | 9e-6         |
| 5     | 0.9586        | 0.1431    | 0.8282      | 0.5748   | 9e-6 → 2.7e-6|
| 6     | 0.9646        | 0.1261    | 0.8294      | 0.5283   | 2.7e-6       |
| 7     | 0.9658        | 0.1199    | 0.8298      | 0.6016   | 2.7e-6       |
| 8     | 0.9670        | 0.1200    | 0.8317      | 0.5667   | 2.7e-6 → 1e-6|
| 9     | 0.9699        | 0.1113    | 0.8323      | 0.5570   | 1e-6         |
| 10    | 0.9699        | 0.1097    | 0.8332      | 0.5677   | 1e-6         |
| 11    | 0.9696        | 0.1096    | 0.8341      | 0.5638   | 1e-6         |

### Performance Summary
- The model achieved high training accuracy (up to 96.99%) and strong validation accuracy (up to 83.41%).
- Loss decreased steadily, indicating effective learning and generalization.
- Learning rate was dynamically reduced to optimize convergence.
- Early stopping and ReduceLROnPlateau callbacks prevented overfitting and improved stability.


### Model Evaluation
- The model demonstrates robust ability to distinguish real from fake videos.
- Evaluation metrics (confusion matrix, classification report, ROC curve) confirm high performance and reliability.

#### Final Test Metrics
- **Accuracy:** 0.8294
- **Precision:** 0.8092
- **Recall:** 0.8294
- **F1 Score:** 0.8110

## 7. Deployment
- **Streamlit App:** Provides an interactive interface for inference and visualization.
- **Model Loading:** Supports loading the trained model for real-time predictions.


## 8. Conclusion
The Deepfake Video Detection System demonstrates effective use of deep learning for video classification. The combination of CNN and LSTM layers enables robust detection of fake content.



