import streamlit as st
import joblib
import numpy as np
import librosa

# Carregar os modelos treinados e o escalador
models = {kernel: joblib.load(f"svm_model_{kernel}.pkl") for kernel in ["linear", "rbf", "poly", "sigmoid"]}
scaler = joblib.load("scaler.pkl")

def extract_features(audio_file):
    try:
        # Carregar o áudio
        y, sr = librosa.load(audio_file, duration=30)  # Limitar a 30 segundos de áudio
        st.write(f"Forma do áudio: {y.shape}, Taxa de amostragem: {sr}")  # Debug
        
        # Extrair MFCC (usando 13 coeficientes)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extrair Chroma (12 coeficientes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Extrair Spectral Contrast (7 coeficientes)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Concatenar as features extraídas
        features = np.hstack([
            np.mean(mfcc, axis=1),            # Média dos coeficientes MFCC
            np.mean(chroma, axis=1),          # Média dos coeficientes Chroma
            np.mean(spectral_contrast, axis=1) # Média dos coeficientes Spectral Contrast
        ])
        
        st.write(f"Features extraídas (antes de reduzir): {features}")  # Debug
        
        # Garantir que temos exatamente 20 features (adaptar conforme necessário)
        features = features[:20]  # Selecionando as 20 primeiras features

        st.write(f"Features finais (após ajuste): {features}")  # Debug
        return features
    
    except Exception as e:
        raise ValueError(f"Erro ao processar o áudio: {e}")


# Interface do Streamlit
st.title("Classificação de Gênero por Voz")
st.write("Envie um arquivo de áudio para prever o gênero usando SVM.")

# Upload do arquivo de áudio
audio_file = st.file_uploader("Carregar arquivo de áudio (.wav)", type=["wav"])

# Se um arquivo for carregado
if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # Seleção do kernel para o modelo
    kernel = st.selectbox("Escolha o kernel do modelo", ["linear", "rbf", "poly", "sigmoid"])
    model = models[kernel]
    st.write(f"Usando modelo SVM com kernel: {kernel}")  # Debug

    if st.button("Fazer Previsão"):
        try:
            # Extrai as features do áudio
            features = extract_features(audio_file)
            
            if len(features) != 20:
                st.error(f"Erro: Número inesperado de features ({len(features)}). Esperado: 20.")
            else:
                # Escalonar as features
                features_scaled = scaler.transform([features])
                st.write(f"Features escalonadas: {features_scaled}")  # Debug

                # Fazer a previsão
                prediction = model.predict(features_scaled)
                gender = "Masculino" if prediction[0] == 0 else "Feminino"

                # Exibir as probabilidades, caso o modelo suporte
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_scaled)
                    st.write(f"Probabilidades: Masculino: {probabilities[0][0]:.2f}, Feminino: {probabilities[0][1]:.2f}")

                st.success(f"Gênero classificado: **{gender}**")

        except ValueError as e:
            st.error(f"Erro: {
