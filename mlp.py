import numpy as np

ACTIVATIONS = {
    'relu': (lambda x: np.maximum(0, x), lambda x: (x > 0).astype(float)),
    'sigmoid': (lambda x: 1/(1+np.exp(-np.clip(x, -500, 500))), 
                lambda x: (s := 1/(1+np.exp(-np.clip(x, -500, 500)))) * (1-s)),
    'softmax': (lambda x: (exps := np.exp(x - np.max(x))) / np.sum(exps), None)
}

LOSS_FUNCTIONS = {
    'mse': (lambda t, p: np.mean((t - p)**2), lambda t, p: 2 * (p - t) / t.size),
    'cross_entropy': (lambda t, p: -np.sum(t * np.log(np.clip(p, 1e-15, 1.0))), 
                      lambda t, p: p - t) 
}


class SGD:
    def update(self, layer, lr):
        layer.weights -= lr * (layer.w_grad / layer.batch_counter)
        layer.biases -= lr * (layer.b_grad / layer.batch_counter)

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m_w, self.v_w = 0, 0
        self.m_b, self.v_b = 0, 0
        self.t = 0

    def update(self, layer, lr):
        self.t += 1
        dw, db = layer.w_grad / layer.batch_counter, layer.b_grad / layer.batch_counter
        
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

OPTIMIZERS = {
    'sgd': SGD,
    'adam': Adam
}

class Layer:
    def __init__(self, nb_input, nb_output, activation_name,optimizer_name='adam'):
        self.activation, self.activation_derivative = ACTIVATIONS[activation_name]
        
        self.weights = np.random.uniform(-np.sqrt(6/(nb_input + nb_output)), 
                                         np.sqrt(6/(nb_input + nb_output)), 
                                         (nb_input, nb_output))
        self.biases = np.zeros(nb_output)
        self.w_grad = np.zeros_like(self.weights)
        self.b_grad = np.zeros_like(self.biases)
        self.batch_counter = 0
        
        self.optimizer = OPTIMIZERS[optimizer_name]()

    def forward(self, input_data):
        self.input_data = input_data
        self.node_values = input_data @ self.weights + self.biases
        self.output = self.activation(self.node_values)
        return self.output
    
    def backward(self, output_gradient):
        if self.activation_derivative is None:
            delta = output_gradient
        else:
            delta = output_gradient * self.activation_derivative(self.node_values)
        
        self.w_grad += np.outer(self.input_data, delta)
        self.b_grad += delta
        
        self.batch_counter += 1
        return delta @ self.weights.T
        
    def apply_gradient(self, lr, batch_len):
        self.optimizer.update(self, lr)
        
        self.w_grad.fill(0)
        self.b_grad.fill(0)
        self.batch_counter = 0

class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        self.layers = layers
        self.learning_rate = learning_rate
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_derivative):
        grad = loss_derivative
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            
    def learn(self, epochs, batch_size, X, y, loss_name, class_weights=None):
        n_samples = len(X)
        loss_fn, loss_der = LOSS_FUNCTIONS[loss_name]

        print(f"Training on {n_samples} samples...")
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_shf = X[indices]
            y_shf = y[indices]

            total_loss = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X_shf[i : i + batch_size]
                y_batch = y_shf[i : i + batch_size]
                batch_len = len(X_batch)

                for j in range(batch_len):
                    output = self.forward(X_batch[j])

                    current_loss = loss_fn(y_batch[j], output)
                    current_der = loss_der(y_batch[j], output)

                    if class_weights is not None:
                        true_class_idx = np.argmax(y_batch[j])
                        weight = class_weights[true_class_idx]

                        current_loss *= weight
                        current_der *= weight 

                    total_loss += current_loss
                    self.backward(current_der)

                for layer in self.layers:
                    layer.apply_gradient(self.learning_rate, batch_len)

            print(f"Epoch {epoch+1}/{epochs} - Avg Loss: {total_loss/n_samples:.4f}")
"""
nn = NeuralNetwork([
    Layer(784, 128, 'relu'),   
    Layer(128, 64,  'relu'),
    Layer(64, 10,   'softmax')
], learning_rate=0.01)

epochs = 1
batch_size = 1000

nn.learn(epochs, batch_size, X_train, y_train, loss_name='cross_entropy',class_weights=[1,1,1,1,1,1,1,1,1,1])


correct = 0
for i in range(len(X_test)):
    prediction = np.argmax(nn.forward(X_test[i]))
    if prediction == y_test_labels[i]:
        correct += 1

print(f"\nFinal Accuracy: {correct/len(X_test) * 100}%")"""