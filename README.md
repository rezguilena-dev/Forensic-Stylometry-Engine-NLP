# Moteur de Stylométrie Forensique 
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-F7931E?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-000020?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Work_In_Progress-orange?style=for-the-badge)

> **Projet : Moteur de Stylométrie Forensique(NLP)  **
> *Université de Caen Normandie — 2025-2026 (Projet en cours)*

Ce projet représente un moteur de stylométrie forensique développé en **Python**. Il vise à appliquer des techniques de **Traitement Automatique du Langage Naturel (NLP)** à des problématiques concrètes de sécurité et d'investigation numérique.

---

## Cas d'Usage 

Les algorithmes développés dans ce projet sont conçus pour être directement applicables à deux enjeux majeurs :
* **Threat Actor Attribution (CTI) :** Identifier l'auteur d'un manifeste, d'un message de ransomware ou d'un fragment de code malveillant en analysant son empreinte stylistique unique.
* **Prévention des Fuites de Données (DLP) :** Détecter des anomalies dans les communications sortantes ou identifier l'exfiltration de documents internes (plagiat).

---

##  Approche Itérative & Architecture

Le développement suit une approche incrémentale, allant des modèles mathématiques et probabilistes vers les réseaux de neurones profonds.

### 1. Baseline Classique (Modèles Statistiques Explicables)
Implémentation  des algorithmes fondamentaux pour valider une baseline performante "Low-Resource" avant de mobiliser des ressources supplémentaires pour les modèles
complexes:
* **Bag of Words & TF-IDF :** Moteur d'extraction et de pondération des caractéristiques textuelles.
* **Classificateur Naive Bayes :** Modèle probabiliste basé sur la fréquence d'apparition des termes.

### 2. Extension vers l'État de l'Art (Deep Learning & Sémantique)
* **Embeddings Sémantiques (LLM) :** Utilisation d'un modèle BERT multilingue (`paraphrase-multilingual-MiniLM-L12-v2` via `SentenceTransformer`) pour encoder le corpus en vecteurs denses et capturer le contexte.
* **Réseau de Neurones (MLP) :** Développement à bas niveau d'un Perceptron Multicouche (fonctions d'activation, rétropropagation, optimiseurs Adam/SGD) entraîné sur les vecteurs générés.

---

##  Évaluation des Performances
L'évaluation se concentre sur les métriques clés en détection d'anomalies et classification :
* **Précision (Precision) :** Limiter les faux positifs (ex: accuser à tort un utilisateur dans un contexte DLP).
* **Rappel (Recall) :** Limiter les faux négatifs (ex: rater l'attribution d'un attaquant connu).
* **F1-Score & Accuracy :** Mesure  globale de la robustesse du moteur.

---

## Organisation et Arborescence: 
   ```
.
├── Main.py               # Orchestrateur d'entraînement et d'évaluation
├── naive_bayes.py        # Algorithme Bayésien Naïf codé à bas niveau
├── tfIdf.py              # Extraction TF-IDF implémentée manuellement
├── Classifier.py         # Modèle de classification de référence (Scikit-Learn)
├── mlp.py                # Moteur de réseau de neurones 
├── Metrique.py           # Calcul des matrices de confusion (Précision/Rappel)
├── encoder.py            # Génération d'embeddings vectoriels (BERT)
└── data-plagiarism/      # Contient le corpus textuel 
   ```
### Équipe de Développement
    Lena REZGUI
    Aris BENSADI
    Mohamed CHAIB SETTI
    Mohamed Yassine LAMAIRI
