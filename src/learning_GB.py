"""Model to train Gradient Bossting model."""

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from utils.constants import PATH_TO_DATA_FOLDER, PATH_TO_DNN_MODEL
import glob

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

    # Initialize the base regressor (GradientBoostingRegressor)
    base_regressor = GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.01, max_depth=3
    )

    # Wrap the base regressor in MultiOutputRegressor for multivariate regression
    multi_output_gbr = MultiOutputRegressor(base_regressor)

    # Perform K-fold cross-validation
    k = 5  # Number of folds
    scores = cross_val_score(
        estimator=MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=3)
        ),
        X=X,
        y=y,
        cv=k,
        scoring="r2",
    )

    # Print the cross-validation scores
    print(f"K-fold cross-validation scores (R^2) for each fold: {scores}")
    print(f"Mean R^2 score: {scores.mean()}")

    # train model
    multi_output_gbr.fit(X, y)

    # save results
    model_filename = PATH_TO_DNN_MODEL.joinpath(camera_name).joinpath(
        f"{camera_name}.joblib"
    )
    joblib.dump(multi_output_gbr, model_filename)

    print(f"Model saved to {model_filename}")
