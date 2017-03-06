
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import math

get_ipython().magic('pylab inline')


# ### Base Random Forest Model

# In[2]:

# EDA
data = pd.read_csv("breast_cancer.csv")
data.head()


# In[3]:

# Remove trivial features & setting predict feature
data = data.drop(['Unnamed: 0', 'id number'], axis=1)
y = data.pop("malignant")
data.describe()


# In[4]:

# Split Test and Training Sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.2, random_state=42)


# In[5]:

# Grid Search for hyperparameter optimization
n_estimators = [100,200,300,400,500]
max_features = ['auto', 'sqrt', 'log2']
min_samples_split = [1,2,3,4,5,6,7,8,9,10]

rfc = RandomForestClassifier(n_jobs=1)
# Parameters of pipelines can be set using ‘__’ separated parameter names:
estimator = GridSearchCV(rfc,
                         dict(n_estimators=n_estimators,
                              max_features=max_features,
                              min_samples_split=min_samples_split
                              ), cv=None, n_jobs=-1)


# In[6]:

estimator.fit(X_train, y_train)
best_rfc = estimator.best_estimator_
best_rfc
best_rfc_predict = best_rfc.predict(X_test)


# ### Base Model Performance

# In[7]:

# Accuracy
base_accuracy = accuracy_score(y_test, best_rfc_predict)
print("Accuracy: ", base_accuracy)


# In[8]:

# Precision & Recall
print(classification_report(y_test, best_rfc_predict))


# In[9]:

# AUC
base_roc = roc_auc_score(y_test, best_rfc.predict_proba(X_test)[:,1])
print("AUC Score: ", base_roc)


# In[10]:

fpr, tpr, thresholds = roc_curve(y_test, best_rfc.predict_proba(X_test)[:,1])
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % base_roc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# ### K-Fold Cross Validation Model

# In[11]:

from sklearn import cross_validation
scores = cross_validation.cross_val_score(best_rfc, data, y, cv=10)
scores


# In[12]:

mean_score = scores.mean()
std_dev = scores.std()
std_error = scores.std() / math.sqrt(scores.shape[0])
ci =  2.262 * std_error
lower_bound = mean_score - ci
upper_bound = mean_score + ci

print ("Score is %f +/- %f" % (mean_score, ci))
print ('95 percent confidence interval for Score is %f and %f' % (lower_bound, upper_bound))


# ### Conclusion and Final Analysis
# > Base Model (Single Holdout AUC)
# + AUC Score:  0.995087719298
# 
# > K-Fold Cross Validation
# + Cross Validation Score: 0.970059 +/- 0.017827 or (0.952233, 0.987886)
# 
# Clearly the AUC from the Single Holdout model quantifies are higher performance if directly with the Cross Validation Score (if those two are even directly comparable); however, the base model may suffer from overfitting indicated by suspiciously high perforance in Accuracy.  It is possible that the 20% holdout used may not be representative of the entire population and/or may not contain specialized edge cases which are critical in determining predictive performance.  By using K-Fold Cross Validation, we minimize the likelihood of this by using the entire dataset to both train and test into an aggregate average score of 0.97 +/- 0.017827
# 
# Digging further into the Precision and Recall figures for the base model, we see that the rate of False Positives most likely unsatisfactory for medical diagnosis purposes; otherwise said, a 3% false diagnosis of malignant is most likely not appropriate and thus we must further refine the model to increase both precision and recall.
