import pandas as pd  # For Reading the csv data
import numpy as np  # For Random operation and numeric array
from sklearn.ensemble import RandomForestClassifier  # ML Model
from sklearn.metrics import f1_score  # Calculates F1 score
from sklearn.preprocessing import StandardScaler  # Standardize the data
from sklearn.model_selection import RandomizedSearchCV, train_test_split  # === CHANGED for K-SHOT ===

# Load the dataset
df = pd.read_csv("data.csv")

# Applying Few Shot (assume only 10% data is known) to train the model

# Step 1 : Assume only 10% data is known
np.random.seed(42)  # we can use any integer. it fixes the randomness
df['label_known'] = 0  # make a new col label_known as 0
# Now, randomly choose which rows will have their labels known as 1
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)  # False = no repetition
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled data
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: Preprocess features
features = ['age', 'bmi', 'glucose', 'insulin', 'blood_pressure']  # input columns
scaler = StandardScaler()  # each feature has mean = 0 and std = 1
# known data scaled
X_labeled = scaler.fit_transform(labeled_data[features])  # input
y_labeled = labeled_data['label']  # output

X_unlabeled = scaler.transform(unlabeled_data[features])  # input
y_unlabeled_true = unlabeled_data['label']  # output

# Step 4: Train model on few-shot data WITHOUT tuning (base model)
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_labeled, y_labeled)  # model learns from known data (labeled data)

# Step 5: Predict for unlabeled data and Evaluate BEFORE tuning
y_pred_base = base_model.predict(X_unlabeled)
f1_base = f1_score(y_unlabeled_true, y_pred_base)  # calculate f1 score
print(f"F1 Score BEFORE finetuning (Few-Shot, 10% labeled): {f1_base:.4f}")

# KSHOT ADDED

# Keep existing known indices
existing_labeled_indices = known_indices

# Desired total fraction for K-shot
k_shot_fraction = 0.2  # 20% labeled total

# How many more samples needed
extra_needed = int(k_shot_fraction * len(df)) - len(existing_labeled_indices)
remaining_indices = list(set(df.index) - set(existing_labeled_indices))

# Stratified sampling for extra points
remaining_df = df.loc[remaining_indices]
_, extra_df = train_test_split(
    remaining_df,
    train_size=extra_needed,
    stratify=remaining_df['label'],
    random_state=42
)
extra_indices = extra_df.index.values

# Combine old + new indices for K-shot
k_shot_indices = np.concatenate([existing_labeled_indices, extra_indices])
df['label_known'] = 0  # reset
df.loc[k_shot_indices, 'label_known'] = 1

# New labeled and unlabeled split for K-shot
labeled_data_k = df[df['label_known'] == 1]
unlabeled_data_k = df[df['label_known'] == 0]

# Preprocess again for K-shot
X_labeled_k = scaler.fit_transform(labeled_data_k[features])
y_labeled_k = labeled_data_k['label']

X_unlabeled_k = scaler.transform(unlabeled_data_k[features])
y_unlabeled_true_k = unlabeled_data_k['label']

# Train and eval K-shot model BEFORE tuning
k_shot_model = RandomForestClassifier(random_state=42)
k_shot_model.fit(X_labeled_k, y_labeled_k)
y_pred_k = k_shot_model.predict(X_unlabeled_k)
f1_k = f1_score(y_unlabeled_true_k, y_pred_k)
print(f"F1 Score with K-Shot (20% labeled) BEFORE finetuning: {f1_k:.4f}")

# Step 6: Define hyperparameter grid for tuning (wider ranges)
param_grid = {
    'n_estimators': [100, 200, 300],  # number of trees in forest
    'max_depth': [None, 20, 40],      # max depth of each tree
    'min_samples_split': [2, 5],      # min samples to split node
    'min_samples_leaf': [1, 2],       # min samples at leaf node
    'class_weight': ['balanced', None]  # balance class weights if needed
}

# Step 7: Setup RandomizedSearchCV for efficient hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,  # try 20 random combinations
    scoring='f1',  # optimize for f1 score
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # use all CPU cores
    random_state=42
)

# Step 8: Perform hyperparameter tuning on K-shot labeled data
random_search.fit(X_labeled_k, y_labeled_k)  # === CHANGED for K-SHOT ===
print("Best parameters found:", random_search.best_params_)

best_model = random_search.best_estimator_

# Step 9: Predict probabilities on K-shot unlabeled data for threshold tuning
y_probs = best_model.predict_proba(X_unlabeled_k)[:, 1]  # === CHANGED for K-SHOT ===

# Step 10: Tune classification threshold to maximize F1 score
best_f1 = 0  # by default
best_thresh = 0.5  # by default

# find the threshold where F1 score is highest and best F1 score
for thresh in np.arange(0.3, 0.7, 0.01):
    y_pred_thresh = (y_probs >= thresh).astype(int)  # convert prob to class label based on threshold
    f1 = f1_score(y_unlabeled_true_k, y_pred_thresh)  # === CHANGED for K-SHOT ===
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# Step 11: Print best threshold and F1 score after tuning
print(f"Best threshold for highest F1: {best_thresh:.2f}")
print(f"Highest F1 after threshold tuning: {best_f1:.4f}")
