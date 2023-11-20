import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


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
new_df = new_df.drop(columns=['Primary_Diagnosis','targets'])


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
lr = LogisticRegression(random_state=42,max_iter = 500)
rf = RandomForestClassifier(random_state=42)
svc = SVC(random_state=42)
ada = AdaBoostClassifier(random_state=42)
knn = KNeighborsClassifier()
voting_clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svc', svc), ("ada", ada), ("knn", knn)], voting='soft')

#------------------Construction of pipeline with preprocessor and classifier from before, and LASSO-based feature selector
X_train, X_test, y_train, y_test = train_test_split(new_df, y, test_size=0.1, random_state=42)

param_grids = {
    'lr': {'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
                            'classifier__penalty': ['l1', 'l2', 'elasticnet', 'none'],
                            'classifier__solver': ['liblinear', 'lbfgs', 'newton-choelsky', 'newton-cg', 'saga']},
                            'classifier__max_iter': [100, 200, 300, 500],
    'knn': {'classifier__n_neighbors': [3, 5, 7, 9],
                            'classifier__weights': ['uniform', 'distance'],
                            'classifier__p': [1, 2]},
    'svc': {'classifier__C': [0.1, 1, 10],
                                  'classifier__kernel': ['linear', 'rbf', 'poly'],
                                  'classifier__degree': [2, 3, 4],
                                  'classifier__gamma': ['scale', 'auto']},
    'ada': {'classifier__n_estimators': [50, 100, 200],
                 'classifier__learning_rate': [0.001, 0.01, 0.1, 1]},
                 'classifier__algorithm': ['SAMME.R', 'SAMME'],
    'rf': {'classifier__n_estimators': [50, 100, 200],
                      'classifier__max_depth': [None, 10, 20],
                      'classifier__criterion': ['gini', 'log_loss', 'entropy'],
                      'classifier__min_samples_split': [2, 5, 10],
                      'classifier__min_samples_leaf': [1, 2, 4]}
}
#---------------------For LogisticRegression


print("################# FOR LR ###############")
param_grid = param_grids['lr']

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', lr)])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
best_params_lr = grid_search.best_params_
print("Best Parameters for LR:", best_params_lr)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#---------------------For RF

print("################# FOR RandomForestClassifier ###############")
param_grid = param_grids['rf']
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', rf)])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params_rf = grid_search.best_params_
print("Best Parameters for RF:", best_params_rf)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#---------------------For SVC

print("################# FOR SVC ###############")
param_grid = param_grids['svc']
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', svc)])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params_svc = grid_search.best_params_
print("Best Parameters for SVC:", best_params_svc)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#---------------------For AdaBoost

print("################# FOR AdaBoost ###############")
param_grid = param_grids['ada']
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', ada)])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params_ada = grid_search.best_params_
print("Best Parameters for SVC:", best_params_ada)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#---------------------For SVC

print("################# FOR KNN ###############")
param_grid = param_grids['knn']
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', knn)])

grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params_knn = grid_search.best_params_
print("Best Parameters for SVC:", best_params_knn)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))






