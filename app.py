import streamlit as st
import joblib
import numpy as np
from audio_recorder_streamlit import audio_recorder
import librosa

# Função para extrair features de um áudio
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features = np.hstack([np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(spectral_contrast, axis=1)])
    return features

# Carregar os modelos SVM e o escalonador
models = {kernel: joblib.load(f"svm_model_{kernel}.pkl") for kernel in ["linear", "rbf", "poly", "sigmoid"]}
scaler = joblib.load("scaler.pkl")

# Interface do Streamlit
st.title("Classificador de Gênero por Voz")
st.write("Envie um arquivo de áudio em formato `.wav` e veja a classificação de gênero.")

# Upload do arquivo de áudio
audio_file = st.file_uploader("Envie um arquivo de áudio (.wav)", type=["wav"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    kernel = st.selectbox("Escolha o kernel do modelo", ["linear", "rbf", "poly", "sigmoid"])
    
    if st.button("Fazer Previsão"):
        try:
            features = extract_features(audio_file)
            features_scaled = scaler.transform([features])
            model = models[kernel]
            prediction = model.predict(features_scaled)
            probabilities = model.predict_proba(features_scaled)
            gender = "Masculino" if prediction[0] == 0 else "Feminino"
            st.success(f"Gênero classificado: **{gender}**")
            st.write(f"Probabilidades: Masculino: {probabilities[0][0]:.2f}, Feminino: {probabilities[0][1]:.2f}")
        except Exception as e:
            st.error(f"Erro ao processar o áudio: {e}")
