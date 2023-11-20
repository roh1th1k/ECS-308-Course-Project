# ECS-308-Course-Project

The given dataset consisted of 775 instances out of which 21 instances contain missing values in at-
least one of the features. In the preprocessing stage, such instances containing missing values along
with their targets were dropped as imputing them reduced the performance of the model. The feature
’Age at diagnosis’ contained string values in the form ’X years Y days’. It was converted to float
values using regular expressions. To analyse the features and see how well they correlate with the
target value, a correlation matrix was constructed. This revealed that the feature ’Primary diagnosis’
is having a perfect correlation of 1 with the target class. Upon further analysis, it was understood
that whenever the primary diagnosis was ’Glioblastoma’, the target class was also found to be GBM
and whenever the primary diagnosis was anything else, the target class was found to be LGG. If a
model is constructed containing this feature, the model would be highly biased to this feature and
would give only negligble weights to the other features. Hence, this feature was dropped. To con-
vert categorical features into numerical values, one-hot-encoding was used and for standardization
of the numerical feature ’Age at diagnosis’, standardscaling was used. For feature selection, a Lin-
earSVC based model, penalized with L1 norm was used. For classification, a voting based approach
based on five classification models was used. Such voting-based methods have been found to perform
better than traditional techniques. A combination of LogisticRegression, k-Nearest Neighbours, Ran-
domForestClassifier, Support Vector Classifier, and AdaBoost with hyperparameters tuned through
GridSearch was used with a soft voting approach. A pipeline was constructed with the preprocessor,
feature selector, and classifier as mentioned before. The training set was split using a regular train
test split technique and the pipeline was fitted. Various performance evaluation metrics like accuracy,
precision, f1-score, recall, and specificity were calculated for all combinations of the 5 classifiers. The
best f1-score is selected.
