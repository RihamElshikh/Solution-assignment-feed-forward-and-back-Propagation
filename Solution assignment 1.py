import numpy as np

class SimpleANN:
    def __init__(self):
        np.random.seed(7)

        self.input_size = 2
        self.hidden_size = 2
        self.output_size = 2

        self.W_input_hidden = np.random.uniform(-0.5, 0.5, (self.input_size, self.hidden_size))
        self.W_hidden_output = np.random.uniform(-0.5, 0.5, (self.hidden_size, self.output_size))

        self.bias_hidden = np.ones((1, self.hidden_size)) * 0.5
        self.bias_output = np.ones((1, self.output_size)) * 0.7

        self.lr = 0.1

    def activation(self, x):
        return np.tanh(x)

    def activation_derivative(self, x):
        return 1 - np.tanh(x)**2

    def forward(self, X):
        self.z_hidden = X @ self.W_input_hidden + self.bias_hidden
        self.a_hidden = self.activation(self.z_hidden)

        self.z_output = self.a_hidden @ self.W_hidden_output + self.bias_output
        self.a_output = self.activation(self.z_output)

        return self.a_output

    def backward(self, X, target):
        error = target - self.a_output

        delta_output = error * self.activation_derivative(self.z_output)
        delta_hidden = (delta_output @ self.W_hidden_output.T) * self.activation_derivative(self.z_hidden)

        self.W_hidden_output += self.a_hidden.T @ delta_output * self.lr
        self.W_input_hidden += X.T @ delta_hidden * self.lr

        self.bias_output += delta_output * self.lr
        self.bias_hidden += delta_hidden * self.lr

    def train(self, X, target, epochs=4000):
        for _ in range(epochs):
            self.forward(X)
            self.backward(X, target)

X_sample = np.array([[0.05, 0.10]])
Y_sample = np.array([[0.01, 0.99]])

model = SimpleANN()
model.train(X_sample, Y_sample)

output = model.forward(X_sample)

print("Predicted Output:")
print(output)