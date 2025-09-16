# IoT-Enabled  Resource Economics  Revealing  America’s  Reliance  on  Minerals  for 
# Sustainable Solutions 
 
 
 
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import (precision_score, accuracy_score, recall_score, f1_score, 
confusion_matrix, classification_report) 
import os, pickle, joblib 
## Importing Dataset 
df = pd.read_csv(r'Datasets/Dataset.csv') 
df 
df.columns 
# EDA 
print("Dataset Info:") 
print(df.info()) 
 
print("\nSummary Statistics:") 
df.describe() 
 
 
df.isnull().sum() 
 
# Correlation heatmap 
plt.figure(figsize=(10, 6)) 
sns.heatmap(df.corr(), annot=True, cmap="coolwarm") 
plt.title("Feature Correlation Heatmap") 
plt.show() 
df.isnull().sum() 
null_percentages = df.isnull().mean() * 100 
 
 
 
print("Percentage of null values in each column:") 
 
 
 
print(null_percentages) 
 
labels = df['sustainable_solution'].unique() 
labels 
X = df.drop(['sustainable_solution'], axis=1) 
X 
y = df['sustainable_solution'] 
y 
y = LabelEncoder().fit_transform(y) 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
# Check the shape of the splits 
 
print("X_train shape:", X_train.shape) 
print("X_test shape:", X_test.shape) 
print("y_train shape:", y_train.shape) 
print("y_test shape:", y_test.shape) 
scaler = StandardScaler() 
 
 
X_train = scaler.fit_transform(X_train) 
 
 
 
X_test = scaler.transform(X_test) 
precision = [] 
recall = [] 
fscore = [] 
accuracy = [] 
 
 
def PerformanceMetrics(algorithm, testY,predict): 
global labels 
testY = testY.astype('int') 
predict = predict.astype('int') 
p = precision_score(testY, predict,average='macro') * 100 
r = recall_score(testY, predict,average='macro') * 100 
f = f1_score(testY, predict,average='macro') * 100 
a = accuracy_score(testY,predict)*100 
accuracy.append(a) 
precision.append(p) 
 
recall.append(r) 
fscore.append(f) 
print(algorithm+' Accuracy  : '+str(a)) 
print(algorithm+' Precision : '+str(p)) 
print(algorithm+' Recall : '+str(r)) 
print(algorithm+' F1-SCORE  : '+str(f)) 
 
 
report=classification_report(predict, testY,target_names=labels) 
print('\n',algorithm+" classification report\n",report) 
conf_matrix = confusion_matrix(testY, predict) 
plt.figure(figsize =(5, 5)) 
ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, 
cmap="Blues" ,fmt ="g"); 
ax.set_ylim([0,len(labels)]) 
plt.title(algorithm+" Confusion matrix") 
plt.ylabel('True class') 
plt.xlabel('Predicted class') 
plt.show() 
labels 
if os.path.exists('model/LinearRegressor.pkl'): 
rge = joblib.load('model/LinearRegressor.pkl') 
print("Model loaded successfully.") 
predict = rge.predict(X_test) 
PerformanceMetrics("Linear Regressor", predict, y_test) 
 
else: 
 
rge = LogisticRegression() #C=1e10, solver='liblinear', class_weight='balanced', 
max_iter=1) 
rge.fit(X_train, y_train) 
 
joblib.dump(rge, 'model/LinearRegressor.pkl') 
print("Model saved successfully.") 
predict = rge.predict(X_test) 
PerformanceMetrics("Linear Regressor", predict, y_test) 
if os.path.exists('model/SVC.pkl'): 
svc_model = joblib.load('model/SVC.pkl') 
print("Model loaded successfully.") 
predict = svc_model.predict(X_test) 
PerformanceMetrics("SVC", predict, y_test) 
else: 
svc_model = SVC() 
svc_model.fit(X_train, y_train) 
# Save the trained model to a file 
joblib.dump(svc_model, 'model/SVC.pkl') 
print("Model saved successfully.") 
predict = svc_model.predict(X_test) 
PerformanceMetrics("SVC", predict, y_test) 
from sklearn.ensemble import RandomForestClassifier 
if os.path.exists('model/RFR.pkl'): 
RFR = joblib.load('model/RFR.pkl') 

print("Model loaded successfully.") 
predict = RFR.predict(X_test) 
PerformanceMetrics("RFR", predict, y_test) 
else: 
RFR = RandomForestClassifier() 
RFR.fit(X_train, y_train) 
# Save the trained model to a file 
joblib.dump(RFR, 'model/RFR.pkl') 
print("Model saved successfully.") 
predict = RFR.predict(X_test) 
PerformanceMetrics("RFR", predict, y_test) 
# Proposed Model Predication on Test data 
test = pd.read_csv('Datasets/testdata.csv') 
test 
 
predict = rge.predict(test.to_numpy()) 
predict 
test['Predict'] = predict 
test