import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Carregar o dataset
data = pd.read_csv("voice.csv")

# Remover duplicados (se houver)
data = data.drop_duplicates()

# Separar features e rótulos
x = data.drop(columns=['label'])
y = data['label'].apply(lambda x: 1 if x == 'male' else 0)  # Converter rótulos para binário

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizar as features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Listar os kernels que queremos testar
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Treinar o modelo e avaliar para cada kernel
for kernel in kernels:
    # Criar e treinar o modelo
    model = SVC(kernel=kernel, probability=True)
    model.fit(x_train_scaled, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = model.predict(x_test_scaled)


    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Female", "Male"], output_dict=True)
    report["accuracy"] = {"precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": len(y_test)}
    
    print(f"==== RESULTADOS COM KERNEL = {kernel.upper()} ====\n")
    for label, metrics in report.items():
        if isinstance(metrics, dict):  # Para classes ou acurácia
            print(f"{label.capitalize():<10} | " +
                  f"Precision: {metrics['precision']:.2f} | " +
                  f"Recall: {metrics['recall']:.2f} | " +
                  f"F1-Score: {metrics['f1-score']:.2f} | " +
                  f"Support: {metrics['support']}")
        else:  # Para outros valores (não dicionários)
            print(f"{label.capitalize()}: {metrics}")
    print("="*40)  # Separador entre os kernels

    # Salvar o modelo treinado com o respectivo kernel
    joblib.dump(model, f'svm_model_{kernel}.pkl')

# Salvar o modelo treinado e o escalador
joblib.dump(scaler, 'scaler.pkl')
