import numpy as np
from numpy import unravel_index
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from copy import deepcopy


def hierarchical_classification(estimator,
                                X_train,
                                y_train,
                                valid_size=0.1,
                                metric=None,
                                early_stopping=0.01,
                                random_state=None):
    
    i=0
    groups = {}
    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                            y_train,
                                                            stratify=y_train,
                                                            test_size=valid_size,    
                                                            random_state=random_state)
    best_metric = early_stopping
    estimators = []
    while (best_metric >= early_stopping) and i < len(np.unique(y_test)):
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        classes = np.unique(y_test)
        estimators.append(deepcopy(estimator))
        
        if ((metric == f1_score) or (metric == precision_score) or (metric == recall_score))\
            and len(classes) > 2:
            curr_metric = metric(y_test, y_pred, average='weighted')
        else:
            curr_metric = metric(y_test, y_pred)
        best_metric = curr_metric - best_metric
        newcm = cm/cm.sum(axis=1)[:,None]
        np.fill_diagonal(newcm, 0)
        newcm = np.tril(newcm + newcm.T).T
        confused_classes = unravel_index(newcm.argmax(), newcm.shape)
        curr_group = 'group_'+str(i)
        groups[curr_group] = [classes[confused_classes[0]], classes[confused_classes[1]]]
        
        for label in groups[curr_group]:
            y_train.replace(to_replace=label, value=curr_group, inplace=True)
            y_test.replace(to_replace=label, value=curr_group, inplace=True)
            
        print('The most confused classes on iteration {} are: {} and {}'.format(i ,classes[confused_classes[0]], classes[confused_classes[1]]))
        
        print("New classes are: {}".format(np.unique(y_test)))
        print("Best {} on validation set is {}".format(metric.__name__, curr_metric))
        print("-"*60)
        i = i+1
    return estimators, groups