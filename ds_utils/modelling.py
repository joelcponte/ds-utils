

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(pipe, param_grid, X):

    model = RandomizedSearchCV(
        pipe,
        param_grid,
        n_jobs=-1,
        n_iter=30,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=5),
        random_state=42,
        return_train_score=True,
    )
    
    _ = model.fit(X.drop(columns=["class"]), X["class"])
    
    return model

def print_model_results(fitted_model, param_grid):
    return (pd.DataFrame(fitted_model.cv_results_)
        .sort_values("mean_test_score", ascending=False)[
            [f"param_{p}" for p in param_grid.keys()] + 
            ["mean_test_score", "std_test_score", "mean_train_score", "std_train_score"]
        ])


def get_linear_model():
    pipe = Pipeline([
        ("scaler", make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=np.number)),
            (OneHotEncoder(), make_column_selector(dtype_include=object)))),
        ("lr", LogisticRegression(solver="liblinear", random_state=42))
    ])
    return pipe

param_grid = {
    "lr__penalty": ["l2", "l1"],
    "lr__C": np.logspace(-4, 4, 8),
}



from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, precision_recall_curve, roc_curve

## precision / recall curve
precision, recall, _ =  precision_recall_curve(y_test=="good", preds)
ap = average_precision_score(y_test=="good", preds)
PrecisionRecallDisplay(
    precision=precision, recall=recall, average_precision=ap
).plot()


#roc curve
fpr, tpr, _ = roc_curve(y_test=="good", preds)
roc_auc = roc_auc_score(y_test, preds)

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()


### transform target
TransformedTargetRegressor(
            regressor=CatBoostRegressor(
            cat_features=train[features].select_dtypes(object).columns.tolist(),
            random_state=42,
            verbose=False),
            transformer=PowerTransformer(method="box-cox"),
        ))]