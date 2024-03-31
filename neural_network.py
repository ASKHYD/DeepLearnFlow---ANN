import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions, loss_function):
        self.layer_sizes = layer_sizes
        self.activation_functions = activation_functions
        self.loss_function = loss_function

        # Initialize weights and biases for each layer
        self.weights = [np.random.randn(prev, curr) for prev, curr in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(1, curr) for curr in layer_sizes[1:]]

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def feedforward(self, X):
        self.layer_inputs = [X]
        self.layer_outputs = [X]

        # Forward pass through each layer
        for i in range(len(self.weights)):
            layer_input = np.dot(self.layer_outputs[i], self.weights[i]) + self.biases[i]
            activation_function = self.activation_functions[i]
            if activation_function == 'sigmoid':
                layer_output = self.sigmoid(layer_input)
            elif activation_function == 'relu':
                layer_output = self.relu(layer_input)
            self.layer_inputs.append(layer_input)
            self.layer_outputs.append(layer_output)

        return self.layer_outputs[-1]

    def backpropagation(self, X, y, learning_rate, l2_penalty=0.01):
        # Calculate gradients using backpropagation
        gradients_weights = [np.zeros_like(w) for w in self.weights]
        gradients_biases = [np.zeros_like(b) for b in self.biases]

        # Compute error at output layer
        if self.loss_function == 'mse':
            output_error = self.layer_outputs[-1] - y
        elif self.loss_function == 'binary_crossentropy':
            output_error = -(y * np.log(self.layer_outputs[-1]) + (1 - y) * np.log(1 - self.layer_outputs[-1]))

        output_delta = output_error * self.sigmoid_derivative(self.layer_outputs[-1])

        # Compute gradients for output layer
        gradients_weights[-1] = np.dot(self.layer_outputs[-2].T, output_delta)
        gradients_biases[-1] = np.sum(output_delta, axis=0, keepdims=True)

        # Backpropagate error through hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            hidden_error = np.dot(output_delta, self.weights[i + 1].T)
            activation_function_derivative = self.sigmoid_derivative if self.activation_functions[
                                                                            i] == 'sigmoid' else self.relu_derivative
            hidden_delta = hidden_error * activation_function_derivative(self.layer_outputs[i + 1])

            gradients_weights[i] = np.dot(self.layer_outputs[i].T, hidden_delta)
            gradients_biases[i] = np.sum(hidden_delta, axis=0, keepdims=True)

            output_delta = hidden_delta

        # Update weights and biases with gradients
        self.weights = [w - learning_rate * (gw + l2_penalty * w) for w, gw in zip(self.weights, gradients_weights)]
        self.biases = [b - learning_rate * gb for b, gb in zip(self.biases, gradients_biases)]

    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, l2_penalty=0.01,
              early_stopping=True):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 10
        no_improvement_count = 0

        for epoch in range(epochs):
            # Shuffle training data and split into mini-batches
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            for batch_start in range(0, len(X_train), batch_size):
                batch_indices = indices[batch_start: batch_start + batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                # Perform forward pass, backpropagation, and update weights
                self.feedforward(batch_X)
                self.backpropagation(batch_X, batch_y, learning_rate, l2_penalty)

            # Compute training loss
            train_output = self.feedforward(X_train)
            train_loss = np.mean(np.abs(y_train - train_output))
            train_losses.append(train_loss)

            # Compute validation loss
            val_output = self.feedforward(X_val)
            val_loss = np.mean(np.abs(y_val - val_output))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if early_stopping and no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in validation loss.")
                break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Training Loss {train_loss}, Validation Loss {val_loss}")

        return train_losses, val_losses

    def predict(self, X):
        return self.feedforward(X)

    def save_model(self, filename):
        np.savez(filename, layer_sizes=self.layer_sizes, weights=self.weights, biases=self.biases)

    def load_model(self, filename):
        data = np.load(filename)
        self.layer_sizes = data['layer_sizes']
        self.weights = data['weights']
        self.biases = data['biases']

    def plot_losses(self, train_losses, val_losses):
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.show()
