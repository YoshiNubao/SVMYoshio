import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o dataset
data = pd.read_csv("voice.csv")

# Transformar rótulos em binário
data['label'] = data['label'].apply(lambda x: 1 if x == 'male' else 0)

# Selecionar duas features para visualização (você pode escolher outras aqui)
features = ['meanfreq', 'sd']  # Substitua por features presentes no seu dataset
x = data[features].values
y = data['label'].values

# Dividir em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Escalar as features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Listar os kernels a serem usados
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Gerar gráfico para cada kernel
for kernel in kernels:
    # Treinar o modelo SVM
    model = SVC(kernel=kernel)
    model.fit(x_train_scaled, y_train)

    # Gerar um grid para prever valores
    x_min, x_max = x_train_scaled[:, 0].min() - 1, x_train_scaled[:, 0].max() + 1
    y_min, y_max = x_train_scaled[:, 1].min() - 1, x_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predizer usando o grid
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # Criar o gráfico
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(x_train_scaled[:, 0], x_train_scaled[:, 1], c=y_train, s=30, cmap=plt.cm.coolwarm, edgecolors='k', label="Treino")
    plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=y_test, s=30, cmap=plt.cm.coolwarm, edgecolors='k', marker='x', label="Teste")
    plt.title(f"Gráfico de Dispersão com Linha de Decisão - Kernel {kernel.capitalize()}")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.show()