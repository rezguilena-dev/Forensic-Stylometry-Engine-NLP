import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Fonctions d'activation et dérivées ---
ACTIVATIONS = {
    'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
    'softmax': (lambda x: (exps := np.exp(x - np.max(x))) / np.sum(exps), None)  
}

# --- Fonctions de perte ---
LOSS_FUNCTIONS = {
    'cross_entropy': (
        lambda t, p: -np.sum(t * np.log(np.clip(p, 1e-15, 1.0))),
        lambda t, p: p - t  # gradient combiné softmax + cross-entropy
    )
}

# --- Optimiseur Adam ---
class Adam:
    def __init__(self):
        self.m_w, self.v_w = None, None
        self.m_b, self.v_b = None, None
        self.t = 0
        self.beta1, self.beta2, self.epsilon = 0.9, 0.999, 1e-8

    def update(self, layer, lr):
        self.t += 1
        dw = layer.w_grad / layer.batch_counter
        db = layer.b_grad / layer.batch_counter

        if self.m_w is None:
            self.m_w, self.v_w = np.zeros_like(dw), np.zeros_like(dw)
            self.m_b, self.v_b = np.zeros_like(db), np.zeros_like(db)

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw**2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db**2)

        m_w_hat = self.m_w / (1 - self.beta1**self.t)
        m_b_hat = self.m_b / (1 - self.beta1**self.t)
        v_w_hat = self.v_w / (1 - self.beta2**self.t)
        v_b_hat = self.v_b / (1 - self.beta2**self.t)

        layer.weights -= lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        layer.biases -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

# --- Classe Layer ---
class Layer:
    def __init__(self, nb_input, nb_output, activation_name):
        self.activation, self.activation_derivative = ACTIVATIONS[activation_name]
        # Initialisation Xavier
        limit = np.sqrt(6 / (nb_input + nb_output))
        self.weights = np.random.uniform(-limit, limit, (nb_input, nb_output))
        self.biases = np.zeros(nb_output)

        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.biases)
        self.batch_counter = 0
        self.optimizer = Adam()

    def forward(self, input_data):
        self.input_data = input_data
        self.node_values = input_data @ self.weights + self.biases
        self.output = self.activation(self.node_values)
        return self.output

    def backward(self, output_gradient):
        # delta = gradient * dérivée activation si existante
        delta = output_gradient if self.activation_derivative is None else output_gradient * self.activation_derivative(self.node_values)
        # Utilisation dot pour batch compatible
        if delta.ndim == 1:
            self.w_grad += np.outer(self.input_data, delta)
        else:
            self.w_grad += self.input_data.T @ delta
        self.b_grad += delta if delta.ndim == 1 else np.sum(delta, axis=0)
        self.batch_counter += 1
        return delta @ self.weights.T if delta.ndim == 1 else delta @ self.weights.T

    def apply_gradient(self, lr):
        self.optimizer.update(self, lr)
        self.w_grad.fill(0)
        self.b_grad.fill(0)
        self.batch_counter = 0

# --- Classe NeuralNetwork ---
class NeuralNetwork:
    def __init__(self, layers, lr):
        self.layers = layers
        self.lr = lr

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_der):
        for layer in reversed(self.layers):
            loss_der = layer.backward(loss_der)

    def learn(self, X, y, epochs=15, batch_size=8):
        for _ in range(epochs):
            indices = np.random.permutation(len(X))
            for i in range(0, len(X), batch_size):
                batch_idx = indices[i:i+batch_size]
                for idx in batch_idx:
                    out = self.forward(X[idx])
                    self.backward(LOSS_FUNCTIONS['cross_entropy'][1](y[idx], out))
                for layer in self.layers:
                    layer.apply_gradient(self.lr)

# --- Fonctions utilitaires ---
def get_rep_name(filename):
    return os.path.splitext(filename)[0]

# --- Exécution MLP sur CSV ---
def run_mlp_session(input_csv_names, lr_val, hidden_layers, data_dir, results_dir):
    h_str = "-".join(map(str, hidden_layers))
    print(f"\nMLP : LR={lr_val}, Architecture={h_str}")
    merged_df = None

    ref_df = pd.read_csv(os.path.join(data_dir, input_csv_names[0]))
    idx_train, idx_test = train_test_split(np.arange(len(ref_df)), test_size=0.3, random_state=42, stratify=ref_df['Category'])

    for name in input_csv_names:
        csv_path = os.path.join(data_dir, name)
        if not os.path.exists(csv_path): 
            continue
        df = pd.read_csv(csv_path)
        rep_name = get_rep_name(name)

        unique_classes = sorted(df['Category'].unique())
        class_to_int = {c: i for i, c in enumerate(unique_classes)}
        y_onehot = np.eye(len(unique_classes))[df['Category'].map(class_to_int).values]
        X = df.drop(columns=['File', 'Task', 'Category']).values

        it, itest = (train_test_split(np.arange(len(df)), test_size=0.3, random_state=42, stratify=df['Category']) 
                     if len(df) != len(ref_df) else (idx_train, idx_test))

        # Création du MLP
        layers = []
        prev_size = X.shape[1]
        for h_size in hidden_layers:
            layers.append(Layer(prev_size, h_size, 'relu'))
            prev_size = h_size
        layers.append(Layer(prev_size, len(unique_classes), 'softmax'))

        nn = NeuralNetwork(layers, lr=lr_val)
        nn.learn(X[it], y_onehot[it])

        # Prédictions et précision
        preds = [unique_classes[np.argmax(nn.forward(X[j]))] for j in itest]
        acc = accuracy_score(df['Category'].iloc[itest], preds)
        print(f"   > {rep_name.ljust(15)} : {acc*100:.2f}%")

        if len(df) == len(ref_df):
            res = pd.DataFrame({
                'File': df['File'].iloc[itest].values,
                'Expected': df['Category'].iloc[itest].values,
                f'Pred_{rep_name}': preds
            })
            merged_df = res if merged_df is None else merged_df.merge(res[['File', f'Pred_{rep_name}']], on='File')

    if merged_df is not None:
        fname = f"MLP_results_LR{lr_val}_Arch{h_str}.csv"
        merged_df.to_csv(os.path.join(results_dir, fname), index=False)
        print(f" Fichier créé : resultats/{fname}")

# --- Boucle sur toutes les architectures et learning rates ---
def run_all_loop(input_csv_names):
    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    project_root = os.path.join(base_dir, '..')
    data_dir = os.path.join(project_root, 'data')
    results_dir = os.path.join(project_root, 'resultats')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    arch_list = [[64], [128, 64]]
    for lr in [0.01, 0.001]:
        for arch in arch_list:
            run_mlp_session(input_csv_names, lr, arch, data_dir, results_dir)

# --- Exécution depuis ligne de commande ---
if __name__ == "__main__":
    fichiers = sys.argv[1:]
    run_all_loop(fichiers)