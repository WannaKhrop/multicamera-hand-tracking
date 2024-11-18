"""Module to train Multi-Layer Neural Network"""

import numpy as np
import glob
from keras import layers, models, regularizers, optimizers
from sklearn.model_selection import KFold
from utils.constants import PATH_TO_DATA_FOLDER, PATH_TO_DNN_MODEL


# Define the neural network model
def create_model(in_shape: int, out_shape: int):
    model = models.Sequential()

    # Input layer
    model.add(layers.InputLayer(input_shape=(in_shape,)))

    # Hidden Layer
    model.add(
        layers.Dense(
            256,
            activation="tanh",
            kernel_regularizer=regularizers.L2(l2=1e-4),
            use_bias=False,
        )
    )
    model.add(layers.Dropout(rate=0.2))

    # Output layes
    model.add(layers.Dense(out_shape, kernel_regularizer=regularizers.L2(l2=1e-3)))
    model.add(layers.LeakyReLU(alpha=0.2))

    # Compile the model using MAE + LOGCOSH as the loss function
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="logcosh",
        metrics=["mae", "logcosh"],
    )

    return model


features = glob.glob(str(PATH_TO_DATA_FOLDER.joinpath("*_features.npy")))
targets = glob.glob(str(PATH_TO_DATA_FOLDER.joinpath("*_targets.npy")))

features.sort()
targets.sort()

for features_file, targets_file in zip(features, targets, strict=True):
    # camera
    camera_name = targets_file.split("\\")[-1].split("_")[0]
    print(f"Camera: {camera_name}")

    # Example data for training (just for demonstration)
    X = np.load(features_file)
    y = np.load(targets_file)

    print("Data shapes: ", X.shape, y.shape)

    # Create the model
    model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])

    # Display the model architecture
    model.summary()

    # Define the number of splits for K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True)  # 10-fold cross-validation

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
            epochs=40,
            batch_size=32,
            verbose=1,
        )

        # Append the validation performance for this fold
        val_mse = history.history["val_mae"][-1]  # Get the last validation MSE
        fold_results.append(val_mse)

    # Calculate the average MSE across all folds
    mean_mae = np.mean(fold_results)
    std_mae = np.std(fold_results)

    print(f"Cross-Validation results: {fold_results}")
    print(f"Cross-Validation results: Mean MAE = {mean_mae}, Std MAE = {std_mae}")

    # Create a new instance of the model
    model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])
    model.fit(X, y, epochs=60, batch_size=32, verbose=1, shuffle=True)

    # save
    model_filename = PATH_TO_DNN_MODEL.joinpath(camera_name).joinpath(
        f"{camera_name}.h5"
    )
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
