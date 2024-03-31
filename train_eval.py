from neural_network import NeuralNetwork
import numpy as np

def train_model(nn, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, l2_penalty=0.01, early_stopping=True):
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
            nn.feedforward(batch_X)
            nn.backpropagation(batch_X, batch_y, learning_rate, l2_penalty)

        # Compute training loss
        train_output = nn.feedforward(X_train)
        train_loss = np.mean(np.abs(y_train - train_output))
        train_losses.append(train_loss)

        # Compute validation loss
        val_output = nn.feedforward(X_val)
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
