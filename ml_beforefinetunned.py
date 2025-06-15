import pandas as pd #For Reading the csv data
import numpy as np #For Random operation and numeric array
from sklearn.ensemble import RandomForestClassifier #ML Model
from sklearn.metrics import f1_score #Calculates F1 score
from sklearn.preprocessing import StandardScaler #Standarize the data
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data.csv")

# Applying Few Shot (assume only 10% data is known) to train the model

#Step1 : Assum only 10% data is known
np.random.seed(42) #we can use any integer. it fix the randomness
df['label_known'] = 0 #make a new col lable_known as 0
#Now,randomly choose which rows will have their labels known as 1
known_indices = np.random.choice(df.index, size=int(0.1 * len(df)), replace=False) #False= As no repetation
df.loc[known_indices, 'label_known'] = 1

# Step 2: Split labeled and unlabeled data
labeled_data = df[df['label_known'] == 1]
unlabeled_data = df[df['label_known'] == 0]

# Step 3: Preprocess features ===
features = ['age', 'bmi', 'glucose', 'insulin', 'blood_pressure'] # input col
scaler = StandardScaler() #each feature has mean = 0 and std = 1 
   #known data scaled
X_labeled = scaler.fit_transform(labeled_data[features]) #input
y_labeled = labeled_data['label'] #output

X_unlabeled = scaler.transform(unlabeled_data[features])  #input
y_unlabeled_true = unlabeled_data['label']  #output

#Step 4: Train model on few-shot data
model = RandomForestClassifier(random_state=42)
model.fit(X_labeled, y_labeled) #model learns from known data i.e labled data

#Step 5: Predict  for 90% unlabled data and Evaluate
y_pred = model.predict(X_unlabeled) #model will predict data in array format based on x_unlabled dayta array which looks like ['age', 'bmi', 'glucose', 'insulin', 'blood_pressure']
f1 = f1_score(y_unlabeled_true, y_pred) #y_unlabeled_true =>lable data in excle sheet and y_pred=> predicted by model

print(f"F1 Score on Unlabeled Data: {f1:.4f}")