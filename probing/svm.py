import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from input_utils import call_input

# Load data
X, y = call_input()

# Parameters
outer_folds = 5
inner_folds = 5
metric = 'roc_auc'
random_seed = 1211

# Hyperparameter grid
param_grid = {
    'pca__n_components': [4, 8, 16, 24, 32, 40],
    'svc__C': [0.01, 0.1, 1, 10, 100],
    'svc__gamma': [1, 0.001, 0.05, 0.075, .1, .5, 1, 5, 10]
}

# Outer CV
outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_seed)
auc_scores, acc_scores = [], []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline([
        ('pca', PCA(whiten=True)),
        ('svc', SVC(kernel='rbf', tol=1e-9))
    ])

    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_seed)
    grid = GridSearchCV(pipeline, param_grid, scoring=metric, cv=inner_cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_score = best_model.decision_function(X_test)

    auc_scores.append(roc_auc_score(y_test, y_score))
    acc_scores.append(accuracy_score(y_test, y_pred))

    print(f"Outer fold AUC: {auc_scores[-1]:.4f}, ACC: {acc_scores[-1]:.4f}, Best Params: {grid.best_params_}")

# Summary
print(f"\Mean CV Results:")
print(f"Mean AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Mean ACC: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
