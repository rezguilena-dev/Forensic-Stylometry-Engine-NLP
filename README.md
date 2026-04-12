# Moteur de Détection de Plagiat Textuel 
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-F7931E?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-000020?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Finished-brightgreen?style=for-the-badge)

> **Projet : Détection Automatique de Plagiat Textuel & NLP**
> -  Université de Caen Normandie — 2025-2026 

Ce projet implémente un système complet de détection automatique du plagiat capable de distinguer quatre niveaux d'altération textuelle.Il automatise l'analyse de l'intégrité des documents en traitant des formes de plagiat allant de la copie exacte à la paraphrase complexe.

> **Note sur l'évolution de l'architecture  :**
> Initialement pensé autour d'embeddings sémantiques profonds (modèles de langage type BERT) pour une approche purement forensique, ce projet a fait l'objet d'une réorientation technique assumée. L'analyse exploratoire du corpus PAN ayant révélé un volume de données très restreint (95 documents), l'utilisation de très grands modèles (LLMs) présentait un risque critique de sur-apprentissage. 
> L'architecture a donc été optimisée vers des méthodes de vectorisation robustes(TF-IDF, N-grammes) couplées à des classifieurs mathématiquement adaptés aux données déséquilibrées (SVM, CNB), tout en conservant le défi du Deep Learning via l'implémentation d'un réseau de neurones (MLP) entièrement *from scratch*.
---

## Objectifs
Le système est entraîné sur le corpus PAN Plagiarism (95 documents effectifs) pour classifier les textes selon quatre catégories:
* **non :** Texte original (40% du corpus).
* **cut :** Copier-coller direct sans modification.
* **heavy:** Paraphrase forte et restructuration syntaxique.
* **light:** Substitution par synonymes et réécriture partielle.
---

##  Architecture du Pipeline:

Le pipeline est conçu de manière modulaire pour tester 84 configurations (combinaisons de représentations et de classifieurs).

### 1. Stratégies de Représentation Vectorielle
Six méthodes d'extraction de caractéristiques ont été développées pour capturer l'empreinte linguistique:
* **Lexicales :** Bag of Words (BoW) et TF-IDF avec gestion des N-grammes (bigrammes et trigrammes) pour détecter les séquences exactes .
* **Syntaxiques :** Reconnaissance d'Entités Nommées (NER) via spaCy pour analyser la structure factuelle indépendamment du vocabulaire utilisé.
* **Validation :** Générateur de données synthétiques (Blob) créant des signatures parfaitement séparables pour valider unitairement les classifieurs.

### 2. Modèles de Classification
Quatre architectures sont intégrées pour traiter les espaces vectoriels de haute dimension (jusqu'à 21 000 caractéristiques):
* **SVM (Support Vector Machine) :** Utilisation d'un noyau cosinus précalculé pour neutraliser l'impact de la longueur des documents.
* **Complement Naive Bayes (CNB) :**  Modèle probabiliste basé sur la fréquence d'apparition des termes ,spécifiquement choisi pour son efficacité face aux jeux de données déséquilibrés.
* **K-NN(k-nearest neighbors ):** Configuré avec la similarité cosinus pour une classification basée sur la proximité spatiale.
* **MLP from scratch :** Réseau de neurones multicouche implémenté entièrement en NumPy (backpropagation, optimiseur Adam, Softmax stable) .
---
## Robustesse et Intégrité des Données:
Face à un dataset réduit et déséquilibré, le pipeline intègre des mécanismes critiques pour limiter les faux négatifs :
* **Échantillonnage stratifié :** Garantit la conservation des proportions de classes lors du split 70/30 .
* **Pondération des classes (class_weight) :** Pénalise plus fortement les erreurs sur les classes minoritaires.
* **Split de référence unique :** Assure la comparabilité absolue des résultats entre toutes les représentations.

---

##  Évaluation des Performances
L'évaluation se concentre sur les métriques clés en détection d'anomalies et classification :
* **Rappel (Recall) :** Mesure la capacité du moteur à identifier tous les cas de plagiat réels (minimise les faux négatifs par exemple un plagiat non détecté).
* **F1-Score Macro :** Moyenne de la précision et du rappel sur toutes les classes (pénalise les déséquilibres et reflète la performance réelle sur les classes minoritaires).
* **Accuracy :** Proportion globale de documents correctement classifiés sur l'ensemble du corpus de test.
  
La configuration la plus performante identifiée est le **SVM** couplé à **TF-IDF N-grammes (C=10)**:
* **Accuracy:** 0.69.
* **Recall macro:** 0.62. 
* **F1-Score Macro:** 0.61.

---
## Installation & Utilisation

### Prérequis
Installer manuellement les dépendances :

```bash
pip install scikit-learn numpy pandas spacy matplotlib
python -m spacy download en_core_web_sm
```

### Lancement du pipeline complet:
```bash
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Fonctionnement du pipeline (run_pipeline.sh)
Le script orchestre automatiquement 3 phases séquentielles :

- **Phase 1 — Représentations :** Exécute tous les scripts de `representation/` pour convertir les textes bruts en matrices vectorisées (CSV) stockées dans `data/`.
- **Phase 2 — Classification :** Passe les CSV générés aux quatre modèles de `classifier/` (SVM, k-NN, Naive Bayes, MLP) pour l'entraînement et les prédictions.
- **Phase 3 — Résultats :** Lance `resultats/results.py` pour compiler les scores et générer les graphiques de performances.
  
## Organisation et Arborescence: 
Le projet suit une architecture modulaire , orchestrée par un script bash global :
   ```
.
├── classifier/           # Modèles de classification (svm_.py, knn.py, naivebayes.py, mlp.py)
├── data/                 # Matrices vectorisées générées prêtes pour l'entraînement (CSV)
├── rapport/              # Rapport technique complet (LaTeX/PDF) et ressources graphiques
├── raw_texts/            # Corpus PAN brut (.txt) et fichiers de métadonnées (CSV)
├── representation/       # Scripts de vectorisation (bow.py, tf_idf.py, ner.py, blob.py...)
├── resultats/            # Fichiers de prédictions, script de visualisation (results.py) et graphiques
└── run_pipeline.sh       # Orchestrateur automatisant toute la chaîne de traitement
   ```

  
### Équipe de Développement et répartition des tâches:
   - **Lena REZGUI** — k-NN, BoW, BoW n-grams, Analyse corpus
   - **Aris BENSADI** — MLP from scratch, Générateur Blob, Orchestration Shell
   - **Mohamed CHAIB SETTI** — SVM, TF-IDF, TF-IDF n-grams, Split de référence
   - **Mohamed Yassine LAMAIRI** — Naive Bayes, NER, Visualisation des résultats
