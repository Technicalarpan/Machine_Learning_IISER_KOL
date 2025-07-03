import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("data.csv")

# Step 1: Simulate Few-Shot (only 10% labeled data)
np.random.seed(42)
df['label_known'] = 0
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False)
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: Vectorize the 'text' column
vectorizer = TfidfVectorizer(max_features=1000)  # You can tune max_features

X_labeled = vectorizer.fit_transform(labeled_data['text'])
y_labeled = labeled_data['label']

X_unlabeled = vectorizer.transform(unlabeled_data['text'])
y_unlabeled_true = unlabeled_data['label']

# Step 4: Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled)

# Step 5: Predict and Evaluate
y_pred = model.predict(X_unlabeled)
f1 = f1_score(y_unlabeled_true, y_pred)

print(f"F1 Score on Unlabeled Data: {f1:.4f}")
