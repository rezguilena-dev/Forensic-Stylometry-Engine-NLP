"""
Génération de la représentation Bag of Words 
Ce script lit les documents texte ,il les transforme en vecteurs
de fréquences d'apparition de mots (Bag of Words), et sauvegarde le résultat dans un CSV.
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def generate_bag_of_words():
    #On définit les chemins relatifs au script courant.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    src_dir = os.path.join(project_root, 'raw_texts')
    data_dir = os.path.join(project_root, 'data')
    info_file = os.path.join(src_dir, 'file_information.csv')
    output_file = os.path.join(data_dir, 'bow.csv')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(info_file):
        print(f"Erreur : Fichier d'information introuvable dans : {src_dir}")
        return

    df_info = pd.read_csv(info_file)
    # Exclusion des documents originaux 
    before = len(df_info)
    df_info = df_info[df_info['Category'] != 'orig'].reset_index(drop=True)
    skipped = before - len(df_info)
    if skipped:
        print(f"Ignoré : {skipped} document(s) avec Category='orig'")

    documents = []
    valid_indices = []

    print(f"Traitement des fichiers dans : {src_dir}")
    for index, row in df_info.iterrows():
        filepath = os.path.join(src_dir, row['File'])
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())
            valid_indices.append(index)
        else:
            print(f"Fichier ignoré (introuvable) : {row['File']}")

    if not documents:
        print("Aucun document n'a été chargé. Vérifiez les noms dans le CSV.")
        return

    print("Génération de la matrice (CountVectorizer)")
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    bow_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    metadata_df = df_info.iloc[valid_indices].reset_index(drop=True)
    final_df = pd.concat([metadata_df, bow_df], axis=1)
    final_df.to_csv(output_file, index=False)
    print(f"Fichier créé : data/bow.csv")
    print(f"Stats : {final_df.shape[0]} documents | {len(vectorizer.get_feature_names_out())} mots uniques.")


if __name__ == "__main__":
    generate_bag_of_words()