import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import time


print("Chargement du modele BERT ")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

DOSSIER_CORPUS = "data-plagiarism"
FICHIER_INFO = "data-plagiarism/file_information.csv"
FICHIER_SORTIE = "dataset.csv"

def get_embedding(text):
    """
    on utilise BERT pour transformer le texte en vecteur de 384 dimensions.
    """
 
    text = text.replace("\n", " ")
   
    return model.encode(text).tolist()
def lire_fichier(filename):
    path = os.path.join(DOSSIER_CORPUS, filename)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")
        return ""


print(" Lecture des fichiers")
df = pd.read_csv(FICHIER_INFO)

print("generation des vecteurs BERT ")
vectors = []

for index, row in df.iterrows():
    filename = row['File']
    
    print(f"Traitement : {filename} ({index+1}/{len(df)})")
    
    texte = lire_fichier(filename)
    if texte.strip(): 
        vec = get_embedding(texte)
        vectors.append(vec)
    else:
        vectors.append([0.0] * 384)
df['embedding'] = vectors
df.to_csv(FICHIER_SORTIE, index=False)

