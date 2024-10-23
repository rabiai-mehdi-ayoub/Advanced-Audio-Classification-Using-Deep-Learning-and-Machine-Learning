import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib

# Load the CNN1D model
CNN1D_Model = load_model('Audio_Classifier_CNN1D.h5')
# Load the ANN model
ANN_Model = load_model('Audio_Classifier_ANN.h5')
# Load the CNN2D model
CNN2D_Model = load_model('Audio_Classifier_CNN2D.h5')

# Load the LabelEncoder object
le = joblib.load('label_encoder.joblib')


def ANN_Prediction(file_name):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # Get the features
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # Scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # Array of features
    prediction_feature = np.array([feature_scaled])
    # Get the ID of label using argmax
    predicted_vector = np.argmax(ANN_Model.predict(prediction_feature), axis=-1)
    # Get the class label from class ID
    predicted_class = le.inverse_transform(predicted_vector)
    # Display the result
    print("ANN has predicted the class as  --> ", predicted_class[0])


def CNN1D_Prediction(file_name):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # Get the features
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # Scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # Array of features
    prediction_feature = np.array([feature_scaled])
    # Expand dimensions
    final_prediction_feature = np.expand_dims(prediction_feature, axis=2)
    # Get the ID of label using argmax
    predicted_vector = np.argmax(CNN1D_Model.predict(final_prediction_feature), axis=-1)
    # Get the class label from class ID
    predicted_class = le.inverse_transform(predicted_vector)
    # Display the result
    print("CNN1D has predicted the class as  --> ", predicted_class[0])

def CNN2D_Prediction(file_name):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    # Get the features
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
    # Scale the features
    feature_scaled = np.mean(feature.T, axis=0)
    # Array of features
    prediction_feature = np.array([feature_scaled])
    # Reshape the features
    final_prediction_feature = prediction_feature.reshape(
        prediction_feature.shape[0], 16, 8, 1
    )
    # Get the ID of label using argmax
    predicted_vector = np.argmax(CNN2D_Model.predict(final_prediction_feature), axis=-1)
    # Get the class label from class ID
    predicted_class = le.inverse_transform(predicted_vector)
    # Display the result
    print("CNN2D has predicted the class as  --> ", predicted_class[0])
    
    
def extract_features(file_path, feature_types):
    # Load the audio file
    audio, sr = librosa.load(file_path, res_type='kaiser_fast')

    # Extract features as needed
    all_features = []

    for feature_type in feature_types:
        if feature_type == 'mfcc':
            features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=128)
        elif feature_type == 'chroma':
            features = librosa.feature.chroma_stft(y=audio, sr=sr)
        elif feature_type == 'mel':
            features = librosa.feature.melspectrogram(y=audio, sr=sr)
        elif feature_type == 'contrast':
            features = librosa.feature.spectral_contrast(y=audio, sr=sr)
        elif feature_type == 'tonnetz':
            features = librosa.feature.tonnetz(y=audio, sr=sr)

        # Normalize with the mean
        features_mean = features.mean(axis=1)
        all_features.append(features_mean)

    # Concatenate features if there are multiple
    if len(all_features) > 1:
        concatenated_features = np.concatenate(all_features)
        return concatenated_features
    else:
        return all_features[0]
    
    
def ML_prediction(file_path, model_path, feature_types):
    # Load the model and label encoder from saved files
    model = joblib.load(model_path)
    label_encoder = joblib.load('label_encoder.joblib')

    # Extract features from the audio file
    extracted_features = extract_features(file_path, feature_types)

    # Convert features to a NumPy array
    features_array = np.array([extracted_features])

    # Ensure the array is C-contiguous
    features_array = np.ascontiguousarray(features_array)

    # Make the prediction
    prediction = model.predict(features_array)

    # Convert the prediction to the actual class
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    print("Predicted class: ", predicted_class)

audio_file_path = 'Mp3_Test_Data/Baby.mp3'

ANN_Prediction(audio_file_path)
CNN1D_Prediction(audio_file_path)
CNN2D_Prediction(audio_file_path)
feature = ['mfcc']
ML_prediction(audio_file_path,'Audio_Classifier_SVM.joblib',feature)
ML_prediction(audio_file_path,'Audio_Classifier_KNN.joblib',feature)
ML_prediction(audio_file_path,'Audio_Classifier_GaussianNB.joblib',feature)
