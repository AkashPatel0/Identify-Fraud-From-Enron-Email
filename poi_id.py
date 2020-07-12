#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
features_list = list(data_dict['YEAGER F SCOTT'].keys())

#store the features in a numpy array for easier manipulation
features = []
for k, v in data_dict.iteritems():
    features.append(v.values())
features = np.array(features)

### Task 3: Create new feature(s)
new_feat = []
for row in features:
    n_nan = len(np.where(row == 'NaN')[0]) # count number of NaN's per row
    new_feat.append(n_nan)
features_list.append('missing data')
features_new = np.hstack([features, np.array(new_feat).reshape(-1,1)]) #concatenate the features with the new feature
for i, name in enumerate(list(data_dict.keys())):
    data_dict[name]['missing data'] = new_feat[i] #include it in the dictionary

#remove both poi label and email address, since it's a string. 
label_index = features_list.index('poi')
email_index = features_list.index('email_address')

labels = features_new[:,label_index]
features_new = np.delete(features_new,[label_index, email_index], axis=1) #delete relevant rows from the features array

features_list = np.delete(features_list, [label_index, email_index]).tolist() #delete from the features list
features_new = features_new.astype(np.float32) # recast it as a float. Any 'NaN' will be turned to np.nan

### Task 2: Remove outliers
features_new = np.nan_to_num(features_new) #turn nans to 0
perc95 = np.percentile(features_new, 95, axis=1) #take the 95th percentile of each column
outliers = []
for i, line in enumerate(features_new):
    for j, value in enumerate(line):
        if value > perc95[j]:
            outliers.append(i) #append to list if a given row is above the desired percentile

#count the number of features each person has above 95th
counts = {}
for idx in outliers:
    if idx in counts.keys():
        counts[idx] += 1
    else:
        counts[idx] = 1
#select those to remove
to_remove = [k for k in counts.keys() if counts[k] > 7]
data_keys = list(data_dict.keys())
for i in to_remove:
    data_dict.pop(data_keys[i])  #delete names from dict
features_sel = np.delete(features_new, to_remove, axis=0) #delete rows from feature array

#select which features to keep
cols_to_keep = []
for i in range(features_sel.shape[1]):
    if np.count_nonzero(features_sel[:,i]) > 70: #it has above 70 non-missing values
        cols_to_keep.append(i)
kept_features = []
for i in cols_to_keep:
    kept_features.append(features_list[i]) #take the relevant feature names

features_final = features_new[:,cols_to_keep]  #rearrange the array


### Store to my_dataset for easy export below.
my_dataset = data_dict
kept_features.insert(0, 'poi')  #inset poi at position 0 as requested
features_list = kept_features
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(class_weight='balanced')



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy import stats
cv = StratifiedShuffleSplit(labels_train, n_iter=20, test_size=0.2) #use a shuffle split to optimize
pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', rf)])
distribution = {'clf__max_depth':[3, 4, 5, 6], 'clf__n_estimators':[40, 80, 115, 120, 150], 'clf__min_samples_split':[2, 3, 4, 5, 6]}
rf_random = GridSearchCV(pipe, distribution, cv=cv, scoring='f1', n_jobs=4)

rf_fit = rf_random.fit(features_train, labels_train)

clf = rf_fit.best_estimator_
#print(clf.named_steps['clf'].feature_importances_, kept_features)
#cv1 = StratifiedKFold(5)
#recall = []
#precision = []
#f1_score = []
#for _ in range(10):
#    score_r = cross_val_score(clf, features, labels, scoring='recall', cv=cv1)
#    recall.append(np.mean(score_r))
#    score_p = cross_val_score(clf, features, labels, scoring='precision', cv=cv1)
#    precision.append(np.mean(score_p))
#    score_f = cross_val_score(clf, features, labels, scoring='f1', cv=cv1)
#    f1_score.append(np.mean(score_f))



#from sklearn import metrics
#print 'recall = ', np.mean(recall)
#print 'precision = ', np.mean(precision)
#print 'f1 = ', np.mean(f1_score)
clf.fit(features_train, labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
