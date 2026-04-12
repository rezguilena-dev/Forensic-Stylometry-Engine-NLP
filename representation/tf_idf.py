import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_tfidf():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    src_dir = os.path.join(project_root, 'raw_texts')
    data_dir = os.path.join(project_root, 'data')
    info_file = os.path.join(src_dir, 'file_information.csv')
    output_file = os.path.join(data_dir, 'tfidf.csv')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(info_file):
        print(f"Erreur : Fichier introuvable ({info_file})")
        return

    df_info = pd.read_csv(info_file)

    before = len(df_info)
    df_info = df_info[df_info['Category'] != 'orig'].reset_index(drop=True)
    skipped = before - len(df_info)
    if skipped:
        print(f"Ignoré : {skipped} document(s) avec Category='orig'")

    documents = []
    valid_indices = []

    print("Lecture des textes en cours...")
    for index, row in df_info.iterrows():
        filepath = os.path.join(src_dir, row['File'])
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    documents.append(f.read())
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as f:
                    documents.append(f.read())
            valid_indices.append(index)
        else:
            print(f"Fichier ignoré (introuvable) : {row['File']}")

    if not documents:
        print("Aucun document n'a été chargé. Vérifiez les noms dans le CSV.")
        return

    print("Génération de la matrice TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 1),
        max_df=0.6,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(documents)

    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    metadata_df = df_info.iloc[valid_indices].reset_index(drop=True)
    final_df = pd.concat([metadata_df, tfidf_df], axis=1)

    final_df.to_csv(output_file, index=False)

    print(f"\nSuccès !")
    print(f"Fichier créé : data/tfidf.csv")
    print(f"Stats : {final_df.shape[0]} documents | {tfidf_df.shape[1]} mots uniques.")


if __name__ == "__main__":
    generate_tfidf()