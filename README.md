# Advanced Audio Classification Project

## Project Overview

This project tackles the challenge of **audio classification** by applying state-of-the-art techniques from both **Deep Learning (DL)** and **Machine Learning (ML)**. The aim is to classify audio samples into **13 distinct classes** (e.g., Baby, Dog, Helicopter, Rain, etc.) using an array of models including:

- Artificial Neural Networks (ANN)
- 1D Convolutional Neural Networks (CNN1D)
- 2D Convolutional Neural Networks (CNN2D)
- Gaussian Naive Bayes (GaussianNB)
- k-Nearest Neighbors (k-NN)
- Support Vector Machine (SVM)

The project relies on feature extraction from the audio signals using **MFCCs**, **Mel-Spectrogram**, **Chroma**, **Tonnetz**, and others, combined with efficient preprocessing and model architectures to achieve high accuracy and computational efficiency.

---

## Theoretical Foundation

### 1. Feature Extraction

The key to effective audio classification is **feature extraction** from raw audio data. We focus on:

- **Mel-Frequency Cepstral Coefficients (MFCCs)**:

  MFCCs represent the power spectrum of an audio signal. They are calculated by taking the Discrete Cosine Transform (DCT) of the log power spectrum computed on a non-linear **Mel scale** of frequency.

- **Chroma Feature**:

  The **Chroma vector** represents the energy distribution across the 12 distinct pitch classes in the Western music scale. This captures harmonic and pitch-related information.

- **Tonnetz**:

  The **Tonnetz (tonal network)** captures harmonic relationships by representing intervals as vectors in a tonal space, which is useful for music-related audio classification.

These features are stacked to form input vectors for classification models.

---

### 2. Deep Learning Models

#### Artificial Neural Network (ANN)

An **ANN** is a fully connected feed-forward network that learns complex relationships by optimizing weight parameters between layers using backpropagation and gradient descent. For this project:

- **Architecture**: A fully connected network with hidden layers using ReLU activation functions and an output layer using softmax.
  
- **Optimization**: We use **Adam optimizer** to minimize the categorical cross-entropy loss.

#### 1D & 2D Convolutional Neural Networks (CNN1D & CNN2D)

CNNs apply convolution operations that learn spatial hierarchies in the data. For audio classification:

- **1D-CNN** applies convolution across time (temporal features), making it suitable for time-domain analysis of audio signals.

- **2D-CNN** applies convolution across both time and frequency (spectral and temporal features), making it ideal for spectrogram-based representations.

The models use pooling layers to downsample and reduce dimensionality, followed by fully connected layers to produce class predictions.

---

### 3. Machine Learning Models

#### Gaussian Naive Bayes (GaussianNB)

This model assumes that the features are conditionally independent given the class label and follow a Gaussian distribution.

#### k-Nearest Neighbors (k-NN)

The **k-NN** classifier predicts the class of a sample by finding the \(k\)-nearest neighbors in the training set and assigning the majority class.

#### Support Vector Machine (SVM)

The **SVM** seeks to find a hyperplane that maximally separates classes. For non-linearly separable data, we use a **Radial Basis Function (RBF) kernel** to map data into higher-dimensional spaces.

---

## Data Pipeline

1. **Preprocessing**:
   - Convert audio files from MP3 to WAV.
   - Extract features using **Librosa** (MFCC, Chroma, Mel-Spectrogram, etc.).
   - Normalize feature vectors for consistent scaling.

2. **Data Splitting**:
   - Data is split into training (80%) and testing (20%) sets using stratified sampling to preserve class distribution.

3. **Feature Storage**:
   - Feature data is stored in `features.csv` and `features.pkl` for use in model training and evaluation.

---

## Model Training and Evaluation

The models were trained on extracted feature vectors and evaluated based on **accuracy**, **training time**, and **prediction time**. Below is the summarized performance:

| Model     | Accuracy    | Training Time (s) | Prediction Time (s) |
|-----------|-------------|-------------------|---------------------|
| ANN       | 70-75%      | 26.209            | 0.107               |
| CNN1D     | 78-86%      | 101.173           | 0.176               |
| CNN2D     | 79-83%      | 26.809            | 0.096               |
| GaussianNB| 63%         | 0.003             | 0.003               |
| k-NN      | 70%         | 0.003             | 0.056               |
| SVM       | 70%         | 0.029             | 0.046               |

---

## How to Run the Project

### Prerequisites

Ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```
### Feature Extraction
To extract features from the audio dataset, run:
  ```bash
  python feature_extraction.py 
  ```
Otherwise, just run the IPYNB
