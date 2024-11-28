import streamlit as st
import joblib
import numpy as np
import librosa

# Carregar todos os modelos treinados
models = {kernel: joblib.load(f"svm_model_{kernel}.pkl") for kernel in ["linear", "rbf", "poly", "sigmoid"]}
scaler = joblib.load("scaler.pkl")

def extract_features(audio_file):
    try:
        # Carregar o áudio
        y, sr = librosa.load(audio_file, duration=30)
        
        # Extrair as características MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Extrair a média de cada característica
        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spectral_contrast, axis=1)
        ])
        
        # Certificar-se de que estamos retornando 20 features
        return features[:20]  # Limitando para as 20 features
    except Exception as e:
        st.error(f"Erro ao processar o áudio: {e}")
        return None

# Interface do Streamlit
st.title("Classificação de Gênero por Voz")
st.write("Envie um arquivo de áudio para prever o gênero usando SVM.")

# Upload do arquivo de áudio
audio_file = st.file_uploader("Carregar arquivo de áudio (.wav)", type=["wav"])

# Variável para armazenar as features extraídas e o gênero previsto
features = None
gender = None

# Se um arquivo for carregado
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # Extrair as features do áudio sempre que o arquivo for alterado
    features = extract_features(audio_file)
    
    # Verifique as features extraídas
    if features is not None:
        st.write("Features extraídas:", features)
    
    # Adicionar um selectbox para o kernel do modelo
    kernel = st.selectbox("Escolha o kernel do modelo", ["linear", "rbf", "poly", "sigmoid"])
    model = models[kernel]  # Escolher o modelo conforme o kernel selecionado

    # Adicionar um botão para fazer a previsão
    if st.button("Fazer Previsão") and features is not None:
        # Escalonar as features
        features_scaled = scaler.transform([features])
        
        # Fazer a previsão
        prediction = model.predict(features_scaled)
        st.write("Previsão:", prediction)  # Verifique o valor da previsão
        
        gender = "Masculino" if prediction[0] == 0 else "Feminino"
        
        # Probabilidades (se suportadas)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features_scaled)
            st.write(f"Probabilidades: Masculino: {probabilities[0][0]:.2f}, Feminino: {probabilities[0][1]:.2f}")
        
        st.success(f"Gênero classificado: **{gender}**")
    else:
        if features is None:
            st.error("Erro ao processar o áudio. Certifique-se de que o áudio tenha 20 características.")
