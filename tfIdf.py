import os
import pandas as pd
import re
import math
from collections import Counter

df_labels = pd.read_csv("data-plagiarism/file_information.csv")
texts_clean = []
for file in df_labels['File']:
    filepath = os.path.join("data-plagiarism/", file)
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        raw_text = f.read().lower()
        clean_text = re.sub(r"[^\w\s]", " ", raw_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        texts_clean.append(clean_text)

labels = df_labels['Category'].tolist()
categories = set(labels)
total_docs = len(labels)

all_words_voc = Counter()
doc_counts = Counter() 

for text in texts_clean:
    unique_words = set(text.split())
    for word in unique_words:
        doc_counts[word] += 1
    all_words_voc.update(text.split())

idfs = {word: math.log(total_docs / count) for word, count in doc_counts.items()}
vocab_size = len(all_words_voc)

categories_tf = {cat: Counter() for cat in categories}
for text, label in zip(texts_clean, labels):
    categories_tf[label].update(text.split())

category_tfidf_sums = {}
category_weights = {cat: {} for cat in categories}

for cat in categories:
    total_tfidf_weight = 0
    for word, count in categories_tf[cat].items():
        weight = count * idfs[word]
        category_weights[cat][word] = weight
        total_tfidf_weight += weight
    category_tfidf_sums[cat] = total_tfidf_weight

def classify_tfidf(text):
    words = text.split()
    category_scores = {}
    
    prior_log_probs = {cat: math.log(labels.count(cat) / total_docs) for cat in categories}

    for cat in categories:
        score = prior_log_probs[cat]
        denominator = category_tfidf_sums[cat] + vocab_size
        
        for word in words:
            if word in idfs: 
                word_weight = category_weights[cat].get(word, 0)
                score += math.log((word_weight + 1) / denominator)
        
        category_scores[cat] = score
    
    return max(category_scores, key=category_scores.get)

accuracy = 0
for i, label in enumerate(labels):
    prediction = classify_tfidf(texts_clean[i])
    if label == prediction and label != "orig":
        accuracy += 1

print(f"Accuracy : {accuracy / (len(labels)-5):.2%}")