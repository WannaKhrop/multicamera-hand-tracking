"""
Module can help to learn neural network to predict targets from features.

Author: Ivan Khrop
Date: 24.09.2024
"""

from keras import layers, models, regularizers
from sklearn.model_selection import KFold
import numpy as np
from utils.constants import PATH_TO_DATA_FOLDER, PATH_TO_DNN_MODEL
from utils.utils import CustomLoss


# Define the neural network model
def create_model(in_shape: int, out_shape: int):
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(in_shape,)))

    # Hidden layers (you can adjust the number of layers and neurons)
    model.add(
        layers.Dense(512, activation="tanh", kernel_regularizer=regularizers.l2(0.001))
    )  # Hidden layer 1
    model.add(
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))
    )  # Hidden layer 2
    model.add(layers.LeakyReLU(alpha=0.2))

    # Output layer
    model.add(layers.Dense(out_shape))
    model.add(layers.LeakyReLU(alpha=0.1))

    # Compile the model using MSE as the loss function
    model.compile(
        optimizer="adam", loss=CustomLoss(weight=0.01), metrics=["mae", "logcosh"]
    )

    return model


# Example data for training (just for demonstration)
X = np.load(str(PATH_TO_DATA_FOLDER.joinpath("features.npy")))
y = np.load(str(PATH_TO_DATA_FOLDER.joinpath("targets.npy")))

# Create the model
model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])

# Display the model architecture
model.summary()

# Define the number of splits for K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation

# Store validation results
fold_results: list[float] = list()

# Perform K-Fold Cross-Validation

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Training fold {fold+1}/{kf.n_splits}...")

    # Split the data into training and validation sets for this fold
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Create a new instance of the model
    model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])

    # Train the model on the training set and validate on the validation set
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=35,
        batch_size=16,
        verbose=1,
    )

    # Append the validation performance for this fold
    val_mse = history.history["val_mae"][-1]  # Get the last validation MSE
    fold_results.append(val_mse)

# Calculate the average MSE across all folds
mean_mse = np.mean(fold_results)
std_mse = np.std(fold_results)

print(f"Cross-Validation results: {fold_results}")
print(f"Cross-Validation results: Mean MAE = {mean_mse}, Std MAE = {std_mse}")

# Create a new instance of the model
model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])
model.fit(X, y, epochs=50, batch_size=16, verbose=1, shuffle=True)
model.save(PATH_TO_DNN_MODEL)
