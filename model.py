import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

data = pd.read_csv("voice.csv")
data = data.drop_duplicates()

x = data.drop(columns=['label'])
y = data['label'].apply(lambda x: 1 if x == 'male' else 0)  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
kernels = ['linear', 'rbf', 'poly', 'sigmoid']


for kernel in kernels:
    model = SVC(kernel=kernel, probability=True)
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Female", "Male"], output_dict=True)
    report["accuracy"] = {"precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": len(y_test)}
    print(f"==== RESULTADOS COM KERNEL = {kernel.upper()} ====\n")
    for label, metrics in report.items():
        if isinstance(metrics, dict):  
            print(f"{label.capitalize():<10} | " +
                  f"Precision: {metrics['precision']:.2f} | " +
                  f"Recall: {metrics['recall']:.2f} | " +
                  f"F1-Score: {metrics['f1-score']:.2f} | " +
                  f"Support: {metrics['support']}") 
        else:  
            print(f"{label.capitalize()}: {metrics}")
    print("="*40)  
    joblib.dump(model, f'svm_model_{kernel}.pkl')

joblib.dump(scaler, 'scaler.pkl')
