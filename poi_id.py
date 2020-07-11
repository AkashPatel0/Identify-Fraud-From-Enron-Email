{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Akash\\Anaconda3\\envs\\dand\\lib\\site-packages\\sklearn\\cross_validation.py:44: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. Also note that the interface of the new CV iterators are different from that of this module. This module will be removed in 0.20.\n",
      "  \"This module will be removed in 0.20.\", DeprecationWarning)\n",
      "C:\\Users\\Akash\\Anaconda3\\envs\\dand\\lib\\site-packages\\sklearn\\ensemble\\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.\n",
      "  from numpy.core.umath_tests import inner1d\n"
     ]
    }
   ],
   "source": [
    "#!/usr/bin/python\n",
    "\n",
    "import sys\n",
    "import pickle\n",
    "sys.path.append(\"../tools/\")\n",
    "import numpy as np\n",
    "from feature_format import featureFormat, targetFeatureSplit\n",
    "from tester import dump_classifier_and_data\n",
    "\n",
    "### Task 1: Select what features you'll use.\n",
    "### features_list is a list of strings, each of which is a feature name.\n",
    "### The first feature must be \"poi\".\n",
    "#features_list = ['poi']\n",
    "\n",
    "### Load the dictionary containing the dataset\n",
    "with open(\"final_project_dataset.pkl\", \"r\") as data_file:\n",
    "    data_dict = pickle.load(data_file)\n",
    "features_list = list(data_dict['YEAGER F SCOTT'].keys())\n",
    "\n",
    "#store the features in a numpy array for easier manipulation\n",
    "features = []\n",
    "for k, v in data_dict.iteritems():\n",
    "    features.append(v.values())\n",
    "features = np.array(features)\n",
    "\n",
    "### Task 3: Create new feature(s)\n",
    "new_feat = []\n",
    "for row in features:\n",
    "    n_nan = len(np.where(row == 'NaN')[0]) # count number of NaN's per row\n",
    "    new_feat.append(n_nan)\n",
    "features_list.append('missing data')\n",
    "features_new = np.hstack([features, np.array(new_feat).reshape(-1,1)]) #concatenate the features with the new feature\n",
    "for i, name in enumerate(list(data_dict.keys())):\n",
    "    data_dict[name]['missing data'] = new_feat[i] #include it in the dictionary\n",
    "\n",
    "#remove both poi label and email address, since it's a string. \n",
    "label_index = features_list.index('poi')\n",
    "email_index = features_list.index('email_address')\n",
    "\n",
    "labels = features_new[:,label_index]\n",
    "features_new = np.delete(features_new,[label_index, email_index], axis=1) #delete relevant rows from the features array\n",
    "\n",
    "features_list = np.delete(features_list, [label_index, email_index]).tolist() #delete from the features list\n",
    "features_new = features_new.astype(np.float32) # recast it as a float. Any 'NaN' will be turned to np.nan\n",
    "\n",
    "### Task 2: Remove outliers\n",
    "features_new = np.nan_to_num(features_new) #turn nans to 0\n",
    "perc95 = np.percentile(features_new, 95, axis=1) #take the 95th percentile of each column\n",
    "outliers = []\n",
    "for i, line in enumerate(features_new):\n",
    "    for j, value in enumerate(line):\n",
    "        if value > perc95[j]:\n",
    "            outliers.append(i) #append to list if a given row is above the desired percentile\n",
    "\n",
    "#count the number of features each person has above 95th\n",
    "counts = {} \n",
    "for idx in outliers:\n",
    "    if idx in counts.keys():\n",
    "        counts[idx] += 1\n",
    "    else:\n",
    "        counts[idx] = 1\n",
    "#select those to remove\n",
    "to_remove = [k for k in counts.keys() if counts[k] > 7]\n",
    "data_keys = list(data_dict.keys())\n",
    "for i in to_remove:\n",
    "    data_dict.pop(data_keys[i]) #delete names from dict\n",
    "features_sel = np.delete(features_new, to_remove, axis=0) #delete rows from feature array\n",
    "\n",
    "#select which features to keep\n",
    "cols_to_keep = []\n",
    "for i in range(features_sel.shape[1]):\n",
    "    if np.count_nonzero(features_sel[:,i]) > 70: #it has above 70 non-missing values\n",
    "        cols_to_keep.append(i)\n",
    "kept_features = []\n",
    "for i in cols_to_keep:\n",
    "    kept_features.append(features_list[i]) #take the relevant feature names\n",
    "\n",
    "#features_final = features_new[:,cols_to_keep] #rearrange the array\n",
    "\n",
    "\n",
    "### Store to my_dataset for easy export below.\n",
    "my_dataset = data_dict\n",
    "kept_features.insert(0, 'poi') #inset poi at position 0 as requested\n",
    "features_list = kept_features\n",
    "### Extract features and labels from dataset for local testing\n",
    "data = featureFormat(my_dataset, features_list, sort_keys = True)\n",
    "labels, features = targetFeatureSplit(data)\n",
    "\n",
    "### Task 4: Try a varity of classifiers\n",
    "### Please name your classifier clf for easy export below.\n",
    "### Note that if you want to do PCA or other multi-stage operations,\n",
    "### you'll need to use Pipelines. For more info:\n",
    "### http://scikit-learn.org/stable/modules/pipeline.html\n",
    "\n",
    "# Provided to give you a starting point. Try a variety of classifiers.\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rf = RandomForestClassifier(class_weight='balanced')\n",
    "\n",
    "\n",
    "\n",
    "### Task 5: Tune your classifier to achieve better than .3 precision and recall \n",
    "### using our testing script. Check the tester.py script in the final project\n",
    "### folder for details on the evaluation method, especially the test_classifier\n",
    "### function. Because of the small size of the dataset, the script uses\n",
    "### stratified shuffle split cross validation. For more info: \n",
    "### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html\n",
    "\n",
    "# Example starting point. Try investigating other evaluation techniques!\n",
    "from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, StratifiedKFold\n",
    "features_train, features_test, labels_train, labels_test = \\\n",
    "    train_test_split(features, labels, test_size=0.3, random_state=42)\n",
    "\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.cross_validation import StratifiedShuffleSplit\n",
    "from scipy import stats\n",
    "cv = StratifiedShuffleSplit(labels_train, n_iter=20, test_size=0.2) #use a shuffle split to optimize\n",
    "pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', rf)]) #scale to [0,1] first, then classify\n",
    "distribution = {'clf__max_depth':stats.randint(low=2, high = 10), 'clf__n_estimators':stats.randint(low=40, high=200), 'clf__min_samples_split':stats.randint(low=2, high=6)} #hyperparameters distributions\n",
    "rf_random = RandomizedSearchCV(pipe, distribution, cv=cv, random_state=12, scoring='f1', n_jobs=4) #random search to optimize f1 score\n",
    "\n",
    "rf_fit = rf_random.fit(features_train, labels_train) #fit to the training set\n",
    "\n",
    "clf = rf_fit.best_estimator_ #take the best estimator\n",
    "\n",
    "#print(clf.named_steps['clf'].feature_importances_, kept_features) #feature importances\n",
    "\n",
    "#cross-validation\n",
    "#cv1 = StratifiedKFold(5)\n",
    "#recall = []\n",
    "#precision = []\n",
    "#f1_score = []\n",
    "#for _ in range(10): #do the kfolding 10 times. sklearn 0.18 doesn't have repeatedstratifiedkfold\n",
    "#    score_r = cross_val_score(clf, features, labels, scoring='recall', cv=cv1)\n",
    "#    recall.append(np.mean(score_r))\n",
    "#    score_p = cross_val_score(clf, features, labels, scoring='precision', cv=cv1)\n",
    "#    precision.append(np.mean(score_p))\n",
    "#    score_f = cross_val_score(clf, features, labels, scoring='f1', cv=cv1)\n",
    "#    f1_score.append(np.mean(score_f))\n",
    "\n",
    "\n",
    "\n",
    "#from sklearn import metrics\n",
    "#print 'recall = ', np.mean(recall)\n",
    "#print 'precision = ', np.mean(precision)\n",
    "#print 'f1 = ', np.mean(f1_score)\n",
    "clf.fit(features_train, labels_train)\n",
    "\n",
    "### Task 6: Dump your classifier, dataset, and features_list so anyone can\n",
    "### check your results. You do not need to change anything below, but make sure\n",
    "### that the version of poi_id.py that you submit can be run on its own and\n",
    "### generates the necessary .pkl files for validating your results.\n",
    "\n",
    "dump_classifier_and_data(clf, my_dataset, features_list)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "python2.7 Udacity",
   "language": "python",
   "name": "dand"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.17"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}