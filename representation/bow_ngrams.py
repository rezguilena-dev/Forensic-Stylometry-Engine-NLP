"""Génération de la représentation Bag of Words avec N-grammes (BoW + Bigrammes)
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def generate_bow_ngrams():
    #On définit les chemins relatifs au script courant.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    src_dir = os.path.join(project_root, 'raw_texts')
    data_dir = os.path.join(project_root, 'data')
    info_file = os.path.join(src_dir, 'file_information.csv')
    output_file = os.path.join(data_dir, 'bow_ngrams.csv')

    if not os.path.exists(info_file):
        print(f"Erreur : Fichier introuvable ({info_file})")
        return

    df_info = pd.read_csv(info_file)

    #--- Exclusion des documents originaux ---
    before = len(df_info)
    df_info = df_info[df_info['Category'] != 'orig'].reset_index(drop=True)
    skipped = before - len(df_info)
    if skipped:
        print(f"Ignoré : {skipped} document(s) avec Category='orig'")

    documents = []
    valid_indices = []

    print("Lecture des textes en cours")
    for index, row in df_info.iterrows():
        filepath = os.path.join(src_dir, row['File'])
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
            valid_indices.append(index)

    if not documents:
        print("Aucun document n'a été chargé.Vérifiez les noms dans le CSV.")
        return

    print("Génération de la matrice Bag of Words + Bigrammes")
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
    X = vectorizer.fit_transform(documents)
    noms_colonnes = vectorizer.get_feature_names_out()
    bow_ngrams_df = pd.DataFrame(X.toarray(), columns=noms_colonnes)
    metadata_df = df_info.iloc[valid_indices].reset_index(drop=True)
    final_df = pd.concat([metadata_df, bow_ngrams_df], axis=1)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    final_df.to_csv(output_file, index=False)

    print(f"Fichier créé : data/bow_ngrams.csv")
    print(f"Attention, la matrice est énorme : {len(noms_colonnes)} colonnes (mots et paires de mots) !")


if __name__ == "__main__":
    generate_bow_ngrams()