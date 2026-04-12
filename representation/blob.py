import os
import pandas as pd
import numpy as np

def create_smart_blob():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '..')
    data_dir = os.path.join(project_root, 'data')
    output_file = os.path.join(data_dir, 'test_blob.csv')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    categories = ['non', 'light', 'heavy', 'cut']
    n_samples_per_class = 20 
    total_samples = len(categories) * n_samples_per_class

    data_rows = []

    for cat in categories:
        for i in range(n_samples_per_class):
            row = {
                'File': f'doc_{cat}_{i}.txt',
                'Task': 'task_a',
                'Category': cat,
                'mot_original': 0, 
                'mot_copie': 0,
                'mot_synonyme': 0
            }
            
            if cat == 'non':
                row['mot_original'] = np.random.randint(10, 20) 
            elif cat == 'cut':
                row['mot_copie'] = np.random.randint(15, 25)   
            elif cat == 'heavy':
                row['mot_copie'] = np.random.randint(8, 12)    
                row['mot_synonyme'] = np.random.randint(5, 10)  
            elif cat == 'light':
                row['mot_synonyme'] = np.random.randint(10, 15) 
            
            data_rows.append(row)

    df = pd.DataFrame(data_rows)
    df.to_csv(output_file, index=False)

    print(f"Blob généré ! Chaque classe a une signature unique.")

if __name__ == "__main__":
    create_smart_blob()