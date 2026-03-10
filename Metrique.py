from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class Metrique:
    def __init__(self, classifier, texts, labels):
        self.classifier = classifier
        self.texts = texts
        self.labels = labels
        print("Calcul des prédictions...")
        self.predictions = [self.classifier.predict(t) for t in self.texts]
        
    def calculer(self, type_metrique="accuracy"):
        if type_metrique == "accuracy":
            return accuracy_score(self.labels, self.predictions)
        elif type_metrique == "precision":
            return precision_score(self.labels, self.predictions, average='macro', zero_division=0)
        elif type_metrique == "recall":
            return recall_score(self.labels, self.predictions, average='macro', zero_division=0)
        elif type_metrique == "f1":
            return f1_score(self.labels, self.predictions, average='macro', zero_division=0)
        return None

   