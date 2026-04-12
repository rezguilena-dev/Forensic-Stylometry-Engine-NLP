import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score

def generer_graphique(results_dir, model_filter, title):
    if model_filter == "All":
        files = [f for f in os.listdir(results_dir) if 'results' in f and f.endswith('.csv')]
    else:
        files = [f for f in os.listdir(results_dir) if f.startswith(f'{model_filter}_results') and f.endswith('.csv')]
        
    if not files:
        print(f" Aucun fichier trouvé pour '{model_filter}'.")
        return

    all_results = []

    for file in files:
        file_name_clean = file.replace('.csv', '')
        parts = file_name_clean.split('_')
        
        model_name = parts[0]
        param_str = parts[-1]
        if "C" in param_str and "=" not in param_str:
            param_str = param_str.replace("C", "C=")
        elif "K" in param_str and "=" not in param_str:
            param_str = param_str.replace("K", "K=")
            
        file_path = os.path.join(results_dir, file)
        df = pd.read_csv(file_path)
            
        if 'Expected' not in df.columns:
            continue 
            
        pred_cols = [col for col in df.columns if col.startswith('Pred_')]
        
        for col in pred_cols:
            rep_name = col.replace('Pred_', '')
            
            acc = accuracy_score(df['Expected'], df[col])
            f1 = f1_score(df['Expected'], df[col], average='macro', zero_division=0)
            rec = recall_score(df['Expected'], df[col], average='macro', zero_division=0)
            
            if model_filter == "All":
                config_label = f"[{model_name}]\n{rep_name}\n({param_str})"
            else:
                config_label = f"{rep_name}\n({param_str})"
                
            all_results.append({
                'Modèle': model_name,
                'Config': config_label,
                'Accuracy': acc,
                'F1-score': f1,
                'Recall': rec
            })

    full_df = pd.DataFrame(all_results)
    if full_df.empty:
        return
        
    top_6_df = full_df.sort_values(by='F1-score', ascending=False).head(6)

    labels = top_6_df['Config']
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14 if model_filter == "All" else 12, 7))

    rects1 = ax.bar(x - width, top_6_df['Accuracy'], width, label='Accuracy', color='#3498db')
    rects2 = ax.bar(x, top_6_df['F1-score'], width, label='F1-score', color='#e74c3c')
    rects3 = ax.bar(x + width, top_6_df['Recall'], width, label='Recall', color='#2ecc71')

    ax.set_ylabel('Scores (0.0 to 1.0)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.20)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    
    nom_fichier = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    chemin_sauvegarde = os.path.join(results_dir, nom_fichier)
    plt.savefig(chemin_sauvegarde, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {nom_fichier}")

    plt.show()
    
    print(f"\n--- Résumé Top 6 : {model_filter} ---")
    top_6_df_print = top_6_df.copy()
    top_6_df_print['Config'] = top_6_df_print['Config'].str.replace('\n', ' ')
    print(top_6_df_print[['Modèle', 'Config', 'Accuracy', 'F1-score', 'Recall']].to_string(index=False))
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    dossier_resultats = './resultats'  
    
    generer_graphique(dossier_resultats, "SVM", "Top 6 SVM Configurations")
    generer_graphique(dossier_resultats, "KNN", "Top 6 KNN Configurations")
    generer_graphique(dossier_resultats, "NB", "Top 6 Naive Bayes Configurations")
    generer_graphique(dossier_resultats, "All", "Top 6 Modeles (Toutes Configurations Confondues)")