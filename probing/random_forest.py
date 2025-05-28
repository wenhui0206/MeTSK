import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from input_utils import call_input

# Load features and labels
X, y = call_input()
n_splits_outer = 5
n_splits_inner = 5
random_state = 1211
metric = 'roc_auc'

# Define hyperparameter search space
param_grid = {
    'pca__n_components': [8, 16, 32, 40],
    'rf__max_depth': [5, 10, 15, 20, 25, 30]
}

# Set up outer CV
outer_cv = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state)
auc_scores = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Pipeline: PCA + RandomForest
    pipeline = Pipeline([
        ('pca', PCA(whiten=True)),
        ('rf', RandomForestClassifier())
    ])

    # Inner CV: Grid Search
    inner_cv = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=inner_cv,
        scoring=metric, n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Evaluate best model on outer test set
    best_model = grid_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_scores.append(auc)

    print(f"Outer fold AUC: {auc:.4f} | Best params: {grid_search.best_params_}")

# Summary
print(f"\nAverage CV AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
