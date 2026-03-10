import os
import pandas as pd
from Classifier import Classifier
from Metrique import Metrique

csv_path = "data-plagiarism/file_information.csv"
data_dir = "data-plagiarism/"
df = pd.read_csv(csv_path)
df = df[df['Category'] != 'orig'] 
texts = []
for file in df['File']:
    filepath = os.path.join(data_dir, file)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        texts.append(f.read())
labels = df['Category'].tolist()
mon_ia = Classifier()
mon_ia.train(texts, labels)
evaluateur = Metrique(mon_ia, texts, labels)
print("\n--- PERFORMANCES DU MODÈLE ---")
for m in ["accuracy", "precision", "recall", "f1"]:
    score = evaluateur.calculer(m)
    print(f"{m.capitalize()} : {score:.2%}")

