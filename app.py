import streamlit as st
import joblib
import numpy as np
import librosa

# Carregar os modelos e o escalador
models = {kernel: joblib.load(f"svm_model_{kernel}.pkl") for kernel in ["linear", "rbf", "poly", "sigmoid"]}
scaler = joblib.load("scaler.pkl")

def extract_features(audio_file, max_length=20):
    try:
        y, sr = librosa.load(audio_file, duration=30)  # Carregar o áudio com duração de 30s
        st.write(f"Forma do áudio: {y.shape}, Taxa de amostragem: {sr}")  # Debug
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # Extrair MFCCs
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Padding ou corte para garantir que temos exatamente `max_length` MFCCs
        if mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_length]

        # Média das características extraídas
        features = np.hstack([
            np.mean(mfcc, axis=1), 
            np.mean(chroma, axis=1), 
            np.mean(spectral_contrast, axis=1)
        ])

        st.write(f"Features extraídas: {features}")  # Debug
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
    kernel = st.selectbox("Escolha o kernel do modelo", ["linear", "rbf", "poly", "sigmoid"])
    model = models[kernel]
    st.write(f"Usando modelo SVM com kernel: {kernel}")  # Debug

    if st.button("Fazer Previsão"):
        try:
            # Extrair as features do áudio
            features = extract_features(audio_file)
            st.write(f"Forma das features extraídas: {features.shape}")  # Debug

            if len(features) != 20:  # Verificar se temos as 20 features
                st.error(f"Erro: Número inesperado de features ({len(features)}). Esperado: 20.")
            else:
                # Escalonar as features
                features_scaled = scaler.transform([features])
                st.write(f"Features escalonadas: {features_scaled}")  # Debug

                # Fazer a previsão
                prediction = model.predict(features_scaled)
                gender = "Masculino" if prediction[0] == 0 else "Feminino"

                # Probabilidades (se suportadas)
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(features_scaled)
                    st.write(f"Probabilidades: Masculino: {probabilities[0][0]:.2f}, Feminino: {probabilities[0][1]:.2f}")

                st.success(f"Gênero classificado: **{gender}**")

        except ValueError as e:
            st.error(f"Erro: {e}")
        except Exception as e:
            st.error(f"Erro ao processar o áudio: {e}")
