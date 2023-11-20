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


#----------------Datasets are loaded and class labels "GBM" and "LGG" are encoded as 1 and 0:
data_in = pd.read_csv('/home/rohith/RO/Machine Learning ECS308/Final Project/Dataset/training_data.csv')
y = pd.read_csv('/home/rohith/RO/Machine Learning ECS308/Final Project/Dataset/training_data_targets.csv', header=None)
y = y.replace(to_replace=['GBM', 'LGG'],
              value=[1,0])

data_in['targets'] = y

#.....TO REMOVE MISSING VALUES

#----------------Fields with "--" or "not reported" will be replaced with NA in new_df.
new_df = data_in.replace(to_replace=["--"],
                 value=[pd.NA])

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

categorical_features = ['Gender', 'Race', 'Primary_Diagnosis', 'IDH1','TP53','ATRX','PTEN','EGFR','CIC','MUC16','PIK3CA','NF1','PIK3R1','FUBP1','RB1','NOTCH1','BCOR','CSMD3','SMARCA4','GRIN2A','IDH2','FAT4','PDGFRA']
numeric_features = ['Age_at_diagnosis']

#------------------Create transformers for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

#------------------Creating classifiers for vote-based classifier
lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=500, random_state=42)
rf = RandomForestClassifier(max_depth=10, n_estimators=300, random_state=42)
svc = SVC(C=0.1, probability=True, random_state=42)
ada = AdaBoostClassifier(learning_rate=0.1, n_estimators=100, random_state=42)
knn = KNeighborsClassifier(n_neighbors=10, p=2)
voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svc', svc), ("ada", ada), ("knn", knn)], voting='soft')

#------------------Construction of pipeline with preprocessor and classifier from before, and LASSO-based feature selector
pipeline = Pipeline(steps=[('preprocessor', preprocessor),('feature_selector', SelectFromModel(LinearSVC(dual='auto', penalty='l1'))),  ('classifier', voting_clf)])
print(new_df.head())
#------------------Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.1, random_state=42)

#------------------Fitting the pipeline to data
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]


#------------------Calculating the performance scores
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, )
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

results_table = pd.DataFrame({
    'Metric': ['LR+RF+SVM+ADA+KNN'],
    'Accuracy': [accuracy],
    'Precision':[precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'Specificity': [specificity],

})

print(tabulate(results_table, headers='keys', tablefmt='pretty', showindex=False))
