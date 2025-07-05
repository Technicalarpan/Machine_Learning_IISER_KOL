import pandas as pd #For Reading the csv data
import numpy as np #For Random operation and numeric array
from sklearn.ensemble import RandomForestClassifier #ML Model
from sklearn.metrics import f1_score #Calculates F1 score
from sklearn.preprocessing import StandardScaler #Standarize the data
from sklearn.model_selection import RandomizedSearchCV #To find the best model setting 

# Load the dataset
df = pd.read_csv("data.csv")


# Applying Few Shot (assume only 10% data is known) to train the model

#Step1 : Assume only 10% data is known
np.random.seed(42) #we can use any integer. it fix the randomness
df['label_known'] = 0 #make a new col lable_known as 0
#Now,randomly choose which rows will have their labels known as 1
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False) #False= As no repetation
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled data
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: Preprocess features 
features = ['age', 'bmi', 'glucose', 'insulin', 'blood_pressure'] # input col
scaler = StandardScaler() #each feature has mean = 0 and std = 1 
   #known data scaled
X_labeled = scaler.fit_transform(labeled_data[features]) #input
y_labeled = labeled_data['label'] #output

X_unlabeled = scaler.transform(unlabeled_data[features])  #input
y_unlabeled_true = unlabeled_data['label']  #output

#Step 4: Train model on few-shot data WITHOUT tuning (base model)
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_labeled, y_labeled) #model learns from known data i.e labled data

#Step 5: Predict for unlabeled data and Evaluate BEFORE tuning
y_pred_base = base_model.predict(X_unlabeled)
f1_base = f1_score(y_unlabeled_true, y_pred_base) #calculate f1 score
print(f"F1 Score BEFORE finetuning: {f1_base:.4f}")

# Step 6: Define hyperparameter grid for tuning (wider ranges)
param_grid = {
    'n_estimators': [100, 200, 300],  #number of trees in forest
    'max_depth': [None, 20, 40],      #max depth of each tree
    'min_samples_split': [2, 5],      #min samples to split node
    'min_samples_leaf': [1, 2],       #min samples at leaf node
    'class_weight': ['balanced', None] #balance class weights if needed
}

# Step 7: Setup RandomizedSearchCV for efficient hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,  #try 20 random combinations
    scoring='f1', #optimize for f1 score
    cv=3,  #3-fold cross-validation
    n_jobs=-1, #use all CPU cores
    random_state=42
)

# Step 8: Perform hyperparameter tuning on labeled data
random_search.fit(X_labeled, y_labeled)
print("Best parameters found:", random_search.best_params_)

best_model = random_search.best_estimator_

# Step 9: Predict probabilities on unlabeled data for threshold tuning
y_probs = best_model.predict_proba(X_unlabeled)[:, 1] #probability of positive class

# Step 10: Tune classification threshold to maximize F1 score
best_f1 = 0 #by default
best_thresh = 0.5 #By defualt

#find the thresold where f1score is highest and best f1score
for thresh in np.arange(0.3, 0.7, 0.01):
    y_pred_thresh = (y_probs >= thresh).astype(int) #convert prob to class label based on threshold
    f1 = f1_score(y_unlabeled_true, y_pred_thresh)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh

# Step 11: Print best threshold and F1 score after tuning
print(f"Best threshold for highest F1: {best_thresh:.2f}")
print(f"Highest F1 after threshold tuning: {best_f1:.4f}")
