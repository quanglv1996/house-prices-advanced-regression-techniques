import sys

sys.path.append('../')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

from util import reader
from util.reader import Reader
from util.util import write_csv, create_folder

path_dataset = '/home/quanglv/Downloads/MIMIC-III/'
path_scripts = './../scripts'
create_folder(path_scripts)

reader = Reader(path_dataset=path_dataset)
# Split the dataset in two equal parts
X_train, y_train = reader.read_data(name='dataset_all', dataset='statistical_mice')


# Step 2
# tuned_parameters = {'objective': ['binary:logistic'],
#                    'learning_rate': [0.1],
#                    'gamma': [0.1],
#                    'subsample': [0.8],
#                    'colsample_bytree': [0.8],
#                    'min_child_weight': range(1,10,2),
#                    'max_depth': range(1,30,2),
#                    'seed': [1024],
#                    'scale_pos_weight': [1]}


# Step 3
# tuned_parameters = {'objective': ['binary:logistic'],
#                    'learning_rate': [0.1],
#                    'gamma':[i/10.0 for i in range(0,20)],
#                    'subsample': [0.8],
#                   'colsample_bytree': [0.8],
#                   'min_child_weight': [1],
#                    'max_depth': [21],
#                    'seed': [1024],
#                   'scale_pos_weight': [1]}

# #Step 4
# tuned_parameters = {'objective': ['binary:logistic'],
#                    'learning_rate': [0.1],
#                    'gamma':[0.1],
#                    'subsample':[i/10.0 for i in range(6,10)],
#                   'colsample_bytree': [i/10.0 for i in range(6,10)],
#                   'min_child_weight': [1],
#                    'max_depth': [16],
#                    'seed': [1024],
# 	                   'scale_pos_weight': [1]}


# #Step 5
# tuned_parameters = {'objective': ['binary:logistic'],
#                    'learning_rate': [0.1],
#                    'gamma':[0.1],
#                    'subsample':[0.9],
#                   'colsample_bytree': [0.6],
#                   'min_child_weight': [1],
#                    'max_depth': [16],
#                    'seed': [1024],
# 	                   'scale_pos_weight': [1],
# 		 'reg_alpha':[1e-6,1e-4, 1e-2, 1, 100]}

# tuned_parameters = {'objective': ['binary:logistic'],
#                     'learning_rate': [0.1],
#                     'gamma':[0.1],
#                     'subsample':[0.8],
#                    'colsample_bytree': [0.8],
#                    'min_child_weight': [1],
#                     'max_depth': [16],
#                     'seed': [1024],
# 	                   'scale_pos_weight': [1],
# 		 'reg_alpha':[0,1e-8, 1e-7, 1e-6]}

# tuned_parameters = {'objective': ['binary:logistic'],
#                     'learning_rate': [0.01],
#                     'gamma':[0.1],
#                     'subsample':[0.8],
#                    'colsample_bytree': [0.8],
#                    'min_child_weight': [1],
#                     'max_depth': [16],
#                     'seed': [1024],
# 	                   'scale_pos_weight': [1],
# 		 'reg_alpha':[1e-6]}

tuned_parameters = {'objective': ['binary:logistic'],
                    'learning_rate': [0.05],
                    'gamma': [0.1],
                    'subsample': [0.8],
                    'colsample_bytree': [0.8],
                    'min_child_weight': [1],
                    'max_depth': [16],
                    'seed': [1024],
                    'scale_pos_weight': [1],
                    'reg_alpha': [1e-6]}

scores = ['roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(XGBClassifier(n_estimators=500, early_stopping_rounds=20), tuned_parameters, cv=10, scoring=score,
                       n_jobs=-1, verbose=10)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))