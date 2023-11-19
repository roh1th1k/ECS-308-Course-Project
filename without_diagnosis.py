import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tabulate import tabulate
import os

#----------------Datasets are loaded and class labels "GBM" and "LGG" are encoded as 1 and 0:
data_in = pd.read_csv('/home/rohith/RO/Machine Learning ECS308/Final Project/Dataset/training_data.csv')
y = pd.read_csv('/home/rohith/RO/Machine Learning ECS308/Final Project/Dataset/training_data_targets.csv', header=None)
y = y.replace(to_replace=['GBM', 'LGG'],
              value=[1,0])

data_in['targets'] = y

#.....TO REMOVE MISSING VALUES

#----------------Fields with "--" or "not reported" will be replaced with NA in new_df.
new_df = data_in.replace(to_replace=["--", 'not reported'],
                 value=[pd.NA, pd.NA])

#----------------Rows/instances containing atleast one field as NA will be dropped
new_df = new_df.dropna(how='any')

y = new_df['targets']
new_df = new_df.drop(columns=['targets'])


#---------TO CONVERT Age_at_Diagnosis FROM STRING TO FLOAT:

#-----------------Define a regular expression pattern to match the "X years Y days" format
pattern = r'(\d+)\s+years(?:\s+(\d+)\s+days)?'

#-----------------Replace the matched pattern with the desired format
new_df['Age_at_diagnosis'] = new_df['Age_at_diagnosis'].str.replace(pattern, lambda x: x.group(1) if x.group(2) is None else f'{int(x.group(1))}.{str(int(x.group(2))//0.365)[0:-2]}', regex=True).astype(float)

#-----------------Encoding molecular data into numericals by changing MUTATED to 1 and NOT_MUTATED to 0 (or we can encode it with OneHotEncoding which increased the performance scores considerably)

# new_df = new_df.replace(to_replace=['MUTATED', 'NOT_MUTATED', 'Male', 'Female'],
#                         value= [1, 0, 1, 0])

#STANDARDIZATION FOR NUMERICAL DATA AND ENCODING FOR REMAINING CATEGORICAL DATA

categorical_features = ['Gender', 'Race', 'IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1','FUBP1','RB1','NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA']
numeric_features = ['Age_at_diagnosis']

#------------------Create transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

#------------------Creating classifiers for vote-based classifier
lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=500, random_state=42)
rf = RandomForestClassifier(max_depth=10, n_estimators=50, min_samples_split=5, min_samples_leaf=2, random_state=42)
svc = SVC(C=0.1, gamma='scale', degree= 2, kernel='rbf', probability=True, random_state=42)
ada = AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=9, p=2, weights='uniform')
voting_clf = VotingClassifier(estimators=[('lr', lr), ('svc', rf), ('ada', ada),], voting='soft')

#------------------Construction of pipeline with preprocessor and classifier from before, and LASSO-based feature selector
pipeline = Pipeline(steps=[('preprocessor', preprocessor),  ('classifier', voting_clf)])
print(new_df.shape)
#------------------Split data into training and testing setsz
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.1, random_state=42)

#------------------Fitting the pipeline to data
test_data = pd.read_csv("Dataset/test_data.csv")
test_data['Age_at_diagnosis'] = test_data['Age_at_diagnosis'].str.replace(pattern, lambda x: x.group(1) if x.group(2) is None else f'{int(x.group(1))}.{str(int(x.group(2))//0.365)[0:-2]}', regex=True).astype(float)
pipeline.fit(new_df, y)
final_labels = pipeline.predict(test_data)


if os.path.exists("test_labels.txt"):
    os.remove("test_labels.txt")
with open("final_labels.txt", "w") as f:
    for i in final_labels:
        if i == 1:
            f.write("GBM\n")
        elif i == 0:
            f.write("LGG\n")
#------------------Calculating the performance scores
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred, )
# accuracy = accuracy_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_proba)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# specificity = tn / (tn + fp)

# results_table = pd.DataFrame({
#     'Metric': ['LR+RF+SVC+ADA+KNN'],
#     'Accuracy': [accuracy],
#     'Precision':[precision],
#     'Recall': [recall],
#     'F1 Score': [f1],
#     'Specificity': [specificity],

# })

# print(tabulate(results_table, headers='keys', tablefmt='pretty', showindex=False))


""""
lr--0.868421052631579, 0.8888888888888888, 0.8275862068965517
rf--0.8947368421052632, 0.9259259259259259
svc--0.881578947368421, 0.9259259259259259
ada--0.881578947368421, 0.9259259259259259
knn--0.868421052631579, 0.8518518518518519

lr+rf -- 0.881578947368421, 0.9259259259259259
lr+svc --0.881578947368421, 0.9259259259259259
lr+ada -- 0.868421052631579, 0.8888888888888888
lr+knn -- 0.881578947368421, 0.881578947368421
rf+svc -- 0.881578947368421, 0.881578947368421
rf+ada -- 0.8947368421052632, 0.9259259259259259
rf+knn -- 0.881578947368421, 0.9259259259259259
svc+ada -- 0.881578947368421, 0.9259259259259259
svc+knn -- 0.881578947368421, 0.9259259259259259
ada+knn -- 0.881578947368421, 0.9259259259259259

lr+rf+svc--0.881578947368421, 0.9259259259259259
lr+rf+ada--0.881578947368421, 0.9259259259259259
lr+rf+knn--0.881578947368421, 0.9259259259259259
lr+svc+ada--0.881578947368421, 0.9259259259259259
lr+svc+knn--0.881578947368421, 0.9259259259259259
lr+ada+knn--0.881578947368421, 0.9259259259259259
rf+svc+ada--0.881578947368421, 0.9259259259259259
rf+svc+knn--0.881578947368421, 0.9259259259259259
Best-------->rf+ada+knn--0.8947368421052632, 0.9259259259259259
svc+ada+knn--0.881578947368421, 0.9259259259259259

lr+rf+svc+ada -- 0.881578947368421, 0.9259259259259259
lr+rf+svc+knn -- 0.881578947368421, 0.9259259259259259
lr+rf+knn+ada -- 0.881578947368421, 0.9259259259259259
lr+svc+ada+knn -- 0.881578947368421, 0.9259259259259259
rf+svc+ada+knn -- 0.881578947368421, 0.9259259259259259

"""