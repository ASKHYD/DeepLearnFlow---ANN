from neural_network import NeuralNetwork
from train_eval import train_model
import numpy as np

if __name__ == "__main__":
    # Create an instance of the NeuralNetwork class
    nn = NeuralNetwork(layer_sizes=[2, 4, 1], activation_functions=['sigmoid', 'sigmoid'], loss_function='mse')

    # Prepare training and validation data
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])
    X_val = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_val = np.array([[0], [1], [1], [0]])

    # Train the neural network
    train_losses, val_losses = train_model(nn, X_train, y_train, X_val, y_val, epochs=1000, batch_size=2, learning_rate=0.01)

    # Plot training and validation losses
    nn.plot_losses(train_losses, val_losses)
