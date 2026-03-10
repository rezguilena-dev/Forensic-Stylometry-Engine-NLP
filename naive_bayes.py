import os
import pandas as pd
import re
import math
from collections import Counter

# 1️⃣ Load & Filter Data
df_labels = pd.read_csv("data-plagiarism/file_information.csv")
df_labels_task_a = df_labels[df_labels['Task'] == 'd'].copy()

texts_clean = []
for file in df_labels['File']:
    filepath = os.path.join("data-plagiarism/", file)
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        # Preprocessing inline for efficiency
        raw_text = f.read().lower()
        clean_text = re.sub(r"[^\w\s]", " ", raw_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        texts_clean.append(clean_text)

labels = df_labels['Category'].tolist()
categories = set(labels)

total_docs = len(labels)
class_counts = Counter(labels)
print(class_counts)
prior_log_probs = {cat: math.log(count / total_docs) for cat, count in class_counts.items()}

categories_voc = {cat: Counter() for cat in categories}
all_words_voc = Counter()

for text, label in zip(texts_clean, labels):
    words = text.split()
    categories_voc[label].update(words)
    all_words_voc.update(words)

vocab_size = len(all_words_voc)
category_word_totals = {cat: sum(categories_voc[cat].values()) for cat in categories}

def classify(text):
    words = text.split()
    category_scores = {}

    for cat in categories:
        score = prior_log_probs[cat]
        
        denominator = category_word_totals[cat] + vocab_size
        
        for word in words:
            if word in all_words_voc: 
                word_count = categories_voc[cat][word]
                score += math.log((word_count + 1) / denominator)
        
        category_scores[cat] = score
    
    return max(category_scores, key=category_scores.get)
accuracy = 0
for i,label in enumerate(labels):
    prediction = classify(texts_clean[i])
    print(f"Document {i} Actual: {label} | Predicted: {prediction}")
    if label == prediction and label != "orig":
        accuracy+=1
print(f"accuracy : {accuracy / (len(labels)-5)}")
