import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_rep_name(filename):
    return os.path.splitext(filename)[0]

def run_knn_all_reps_for_k(input_csv_names, k_value, data_dir, results_dir):
    """
    Entraîne et évalue un classifieur k-NN pour chaque représentation textuelle,
    avec une valeur de k donnée. Les résultats sont sauvegardés dans un CSV."""

    print(f"\n Génération du rapport complet pour k-NN (k = {k_value})")
    
    merged_df = None
    
    # On définit le split de référence, pour garantir qu'on évalue sur les mêmes documents.
    reference_df = pd.read_csv(os.path.join(data_dir, input_csv_names[0]))
    ref_size = len(reference_df)
    indices = np.arange(ref_size)
    idx_train, idx_test = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=reference_df['Category']
    )
    for input_csv_name in input_csv_names:
        csv_path = os.path.join(data_dir, input_csv_name)
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
       
        # permet la gestion du split si la taille diffère (test_blob par exemple)
        if len(df) != ref_size:
            current_indices = np.arange(len(df))
            c_idx_train, c_idx_test = train_test_split(
                current_indices, test_size=0.3, random_state=42, stratify=df['Category']
            )
        else:
            c_idx_train, c_idx_test = idx_train, idx_test

        rep_name = get_rep_name(input_csv_name)
        y = df['Category']
        noms_fichiers = df['File']
        X = df.drop(columns=['File', 'Task', 'Category'])

        # Entraînement et évaluation du classifieur k-NN 
        modele = KNeighborsClassifier(n_neighbors=k_value, metric='cosine')
        modele.fit(X.iloc[c_idx_train], y.iloc[c_idx_train])
        predictions = modele.predict(X.iloc[c_idx_test])
        acc = accuracy_score(y.iloc[c_idx_test], predictions)

        print(f"   > {rep_name.ljust(15)} : {acc*100:.2f}%")
        if len(df) == ref_size:
            current_res = pd.DataFrame({
                'File': noms_fichiers.iloc[c_idx_test].values,
                'Expected': y.iloc[c_idx_test].values,
                f'Pred_{rep_name}': predictions
            })
            if merged_df is None:
                merged_df = current_res
            else:
                merged_df = merged_df.merge(current_res[['File', f'Pred_{rep_name}']], on='File')


    if merged_df is not None:
        output_filename = f"KNN_results_k{k_value}.csv"
        merged_df.to_csv(os.path.join(results_dir, output_filename), index=False)
        print(f"Rapport global créé : resultats/{output_filename}")

def run_all_loop(input_csv_names):
    """
    Point d'entrée principal : permet de configurer les chemins et lance l'évaluation
    pour chaque valeur de k définie dans la liste.
    """
    #On définit les chemins relatifs au script courant.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'resultats')
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    # --- Valeurs de k testées ---
    for k in [1, 3, 5]:
        run_knn_all_reps_for_k(input_csv_names, k, data_dir, results_dir)

if __name__ == "__main__":
    fichiers = ["bow.csv", "tfidf.csv", "bow_ngrams.csv", "tfidf_ngrams.csv", "ner_bow.csv", "test_blob.csv"]
    
    if len(sys.argv) > 1:
        fichiers = sys.argv[1:]
        
    run_all_loop(fichiers)