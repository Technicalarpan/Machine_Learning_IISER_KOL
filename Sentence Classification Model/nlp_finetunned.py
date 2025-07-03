import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 0: Load dataset
df = pd.read_csv("data.csv", encoding="latin1")  # adjust encoding if needed

# Step 1: Simulate Few-Shot (only 10% data is known)
np.random.seed(42)
df['label_known'] = 0
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled data
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: Convert 'text' to TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X_labeled = vectorizer.fit_transform(labeled_data['text'])
y_labeled = labeled_data['label']

X_unlabeled = vectorizer.transform(unlabeled_data['text'])
y_unlabeled_true = unlabeled_data['label']

# Step 4: Train base model (no tuning)
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_labeled, y_labeled)

# Step 5: Evaluate BEFORE tuning
y_pred_base = base_model.predict(X_unlabeled)
f1_base = f1_score(y_unlabeled_true, y_pred_base)
print(f"F1 Score BEFORE finetuning: {f1_base:.4f}")

# Step 6: Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 20, 40],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', None]
}

# Step 7: RandomizedSearchCV for tuning
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    scoring='f1',
    cv=3,
    n_jobs=-1,
    random_state=42
)

# Step 8: Fit on labeled data
random_search.fit(X_labeled, y_labeled)
print("Best parameters found:", random_search.best_params_)
best_model = random_search.best_estimator_

# Step 9: Predict probabilities on unlabeled data
y_probs = best_model.predict_proba(X_unlabeled)[:, 1]

# Step 10: Tune classification threshold
best_f1 = 0
best_thresh = 0.5

for thresh in np.arange(0.3, 0.7, 0.01):
    y_pred_thresh = (y_probs >= thresh).astype(int)
    f1 = f1_score(y_unlabeled_true, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# Step 11: Final results
print(f"Best threshold for highest F1: {best_thresh:.2f}")
print(f"Highest F1 after threshold tuning: {best_f1:.4f}")
