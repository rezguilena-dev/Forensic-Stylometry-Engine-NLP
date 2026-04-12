import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score

def get_rep_name(filename):
    return os.path.splitext(filename)[0]

def run_nb_all_reps_for_alpha(input_csv_names, alpha_value, data_dir, results_dir):
    print(f"\nGénération du rapport entier pour Naive Bayes (alpha = {alpha_value})")
    
    merged_df = None
    
    # Définition du split de référence 
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
        
        # Gestion du split dans le cas ou la taille diffère (ex: test_blob)
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

        
        X = X.clip(lower=0)

        # Initialisation et entraînement du ComplementNB 
        modele = ComplementNB(alpha=alpha_value)
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
        output_filename = f"NB_results_alpha{alpha_value}.csv"
        merged_df.to_csv(os.path.join(results_dir, output_filename), index=False)
        print(f"Rapport groupé créé : resultats/{output_filename}")

def run_all_loop(input_csv_names):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'resultats')
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    # Liste des valeurs alpha à tester
    for a in [0.01, 0.1, 0.5, 1.0]:
        run_nb_all_reps_for_alpha(input_csv_names, a, data_dir, results_dir)

if __name__ == "__main__":
    fichiers = ["bow.csv", "tfidf.csv", "bow_ngrams.csv", "tfidf_ngrams.csv", "ner_bow.csv", "test_blob.csv"]
    
    if len(sys.argv) > 1:
        fichiers = sys.argv[1:]
        
    run_all_loop(fichiers)