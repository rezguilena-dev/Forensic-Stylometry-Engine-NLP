import os
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

def generate_ner_bow():
    print("Chargement de spaCy")
   
    nlp = spacy.load("en_core_web_sm")
    
    #Définition des chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    src_dir = os.path.join(project_root, 'raw_texts')
    data_dir = os.path.join(project_root, 'data')
    info_file = os.path.join(src_dir, 'file_information.csv')
    output_file = os.path.join(data_dir, 'ner_bow.csv')

    df_info = pd.read_csv(info_file)
    df_suspect = df_info[df_info['Category'] != 'orig'].reset_index(drop=True)

    documents_entities = []
    valid_indices = []

    print("Extraction des entités nommées en cours")
    for index, row in df_suspect.iterrows():
        filepath = os.path.join(src_dir, row['File'])
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            doc = nlp(text)   
            entities = [ent.text.replace(" ", "_") for ent in doc.ents]          
            documents_entities.append(" ".join(entities))
            valid_indices.append(index)

    print("Génération de la matrice Bag of Entities")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents_entities)

    #Sauvegarder dans un nouveau CSV
    ner_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    metadata_df = df_suspect.iloc[valid_indices].reset_index(drop=True)
    final_df = pd.concat([metadata_df, ner_df], axis=1)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    final_df.to_csv(output_file, index=False)
    print(f"\nFichier créé avec succès: data/ner_bow.csv")
    print(f"Le modèle a trouvé {ner_df.shape[1]} entités uniques dans vos textes.")

if __name__ == "__main__":
    generate_ner_bow()