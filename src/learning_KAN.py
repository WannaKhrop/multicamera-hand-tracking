"""Module to train KAN model."""

import numpy as np
import glob
from sklearn.model_selection import KFold
from kan import KAN

# from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import torch

# import joblib
from utils.constants import PATH_TO_DATA_FOLDER, PATH_TO_DNN_MODEL


def logcosh(y_pred, y_true):
    # Calculate the log-cosh loss
    loss = torch.log(torch.cosh(y_pred - y_true))
    return loss.mean()


def mean_absolute_error(y_pred, y_true):
    # Calculate the MAE loss
    loss = torch.abs(y_pred - y_true)
    return loss.mean()


def create_model(in_shape: int, out_shape: int):
    # Create and train the KAN model
    model = KAN(width=[in_shape, 32, out_shape], grid=6, k=3)
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

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Define the number of splits for K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True)  # 10-fold cross-validation

    # Store validation results
    fold_results: dict[str, list[float]] = dict()
    fold_results["kan_loss"] = list()
    fold_results["mae"] = list()
    fold_results["logcosh"] = list()

    # Perform K-Fold Cross-Validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold+1}/{kf.n_splits}...")

        # Split the data into training and validation sets for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dataset = dict()
        dataset["train_input"] = torch.from_numpy(X_train).float()
        dataset["train_label"] = torch.from_numpy(y_train).float()
        dataset["test_input"] = torch.from_numpy(X_val).float()
        dataset["test_label"] = torch.from_numpy(y_val).float()

        # Create the model
        model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])
        history = model.fit(
            dataset=dataset,
            steps=200,
            batch=100,
            opt="LBFGS",
            lr=1e-2,
            log=-1,
        )

        # Make predictions using the loaded model
        y_predict = model(dataset["test_input"])
        fold_results["kan_loss"].append(history["test_loss"][-1])
        fold_results["mae"].append(
            mean_absolute_error(y_pred=y_predict, y_true=dataset["test_label"])
            .detach()
            .numpy()
        )
        fold_results["logcosh"].append(
            logcosh(y_pred=y_predict, y_true=dataset["test_label"]).detach().numpy()
        )

        # plot result
        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["test_loss"], label="Test")
        plt.legend()
        plt.show()

    # evaluation results
    print(fold_results)

    model = create_model(in_shape=X.shape[1], out_shape=y.shape[1])
    dataset = dict()
    dataset["train_input"] = torch.from_numpy(X).float()
    dataset["train_label"] = torch.from_numpy(y).float()
    dataset["test_input"] = torch.from_numpy(X).float()
    dataset["test_label"] = torch.from_numpy(y).float()
    history = model.fit(
        dataset=dataset,
        steps=200,
        batch=100,
        opt="LBFGS",
        lr=1e-2,
        log=-1,
    )
    # plot results
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["test_loss"], label="Test")
    plt.legend()
    plt.show()

    # save
    model_filename = PATH_TO_DNN_MODEL.joinpath(camera_name).joinpath("mark")
    scaler_filename = PATH_TO_DNN_MODEL.joinpath(camera_name).joinpath("scaler.joblib")
    # save model and scaler
    model.prune()
    model.saveckpt(path=str(model_filename))
    # joblib.dump(scaler, scaler_filename)
