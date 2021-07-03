

# Assignment Part 2 

#Outliers have been kept as thereare are few bankrupt companies
#Bankrupt companies may also show outlier financial information

#Initialising packages

# General Libraries


import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

# Machine Learning Libraries

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

#Loading Data
filename = 'data.csv'
Bankruptcy_Data = pd.read_csv(filename)
#Correlation Matrix
plt.figure(figsize=(17,17))
sns.heatmap(Bankruptcy_Data.corr(), annot=False, cmap='coolwarm')
plt.show()

# Dividing Data and Labels

y = Bankruptcy_Data['Bankrupt?']
X = Bankruptcy_Data.drop('Bankrupt?', axis=1)

#Data is imbalanced , so to balance it we will use balancing technique .Here we are using SMOTE only on the training set not the test set 

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=97, test_size=0.2)


from imblearn.over_sampling import SMOTE

over = SMOTE()
X_train, y_train = over.fit_resample(X_train, y_train)



# class imbalance is treated
sns.set_style('white');
sns.set_context(context='notebook',font_scale=1.2)
sns.countplot(x=y_train);
plt.title('Target variable balanced');

#Standarization 
#andardization of a dataset is a common requirement for many machine learning estimators: 
#they might behave badly if the individual features do not more or less look like standard normally distributed data 
#(e.g. Gaussian with 0 mean and unit variance).
#From reviewing certain online resources this helps with PCA

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_train_sc = pd.DataFrame(X_train_sc, columns=X_train.columns, index=X_train.index)

X_test_sc = sc.transform(X_test)
X_test_sc = pd.DataFrame(X_test_sc, columns=X_test.columns, index=X_test.index)


#Principal component analysis and visualization
from sklearn.decomposition import PCA
pc = PCA(n_components=len(X_train_sc.columns))
X_train_pc=pc.fit_transform(X_train_sc)
PC_df_train=pd.DataFrame(X_train_pc,columns=['PC_' +str(i) for i in range(1,pc.n_components_+1)])

PC_df_train

#Scree Plot - PCA Analysis
#In multivariate statistics, a scree plot is a line plot of the eigenvalues of factors or principal components in an analysis. 
#the scree plot is used to determine the number of factors to retain in an exploratory factor analysis (FA) 
#or principal components to keep in a principal component analysis (PCA)
#https://www.kaggle.com/kundankumarmandal/company-bankruptcy-prediction-96-accuracy
plt.figure(figsize=(20,10))
plt.plot(PC_df_train.std())
plt.title('Scree Plot - PCA components')
plt.xlabel('Principal Component')
plt.xticks(rotation=90)
plt.ylabel('Standard deviation')
plt.show()

pc = PCA(n_components=15)
X_train_pc=pc.fit_transform(X_train_sc)
PC_df_train=pd.DataFrame(X_train_pc,columns=['PC_' +str(i) for i in range(1,pc.n_components_+1)])

X_test_pc = pc.transform(X_test_sc)
PC_df_test=pd.DataFrame(X_test_pc,columns=['PC_' +str(i) for i in range(1,pc.n_components_+1)])


#Model Building

print(pc.get_params)

print(PC_df_train.shape)
y_train.shape

#Logistic Regression - Grid search

from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

param_grid = [
    {'C': [1, 10, 100, 1000, 1e4, 1e5, 1e6, 1e7], 'penalty': ['l1', 'l2']}
]
logreg = GridSearchCV(LogisticRegression(), param_grid)
logreg.fit(PC_df_train, y_train)

#print(logreg.cv_results_[{'mean_test_score','std_test_score','params'}])
scoring = logreg.cv_results_
for mean_score, std, params in zip(scoring['mean_test_score'],scoring['std_test_score'],scoring['params']):
    print("{:0.3f} (+/-{:0.03f}) for {}".format(
            mean_score, std * 2, params))

# Print best params
print('\nBest parameters:', logreg.best_params_)


#Suggested parameters = Best parameters: {'C': 10, 'penalty': 'l2'}


#Logistic Regression - Using Grid Search's best parameters

classifier = LogisticRegression(penalty='l2', C=10.0,random_state=42)
classifier.fit(PC_df_train,y_train)
y_lr_GS=classifier.predict(X_test_pc)

print('Confusion Matrix \n',confusion_matrix(y_lr_GS,y_test))
print()
print('Accuracy Score \n', accuracy_score(y_lr_GS,y_test))
print()
print('Classification Report \n',classification_report(y_lr_GS,y_test))



#Logistic Regression without Grid Search Parameters
classifier = LogisticRegression(random_state=42)
classifier.fit(PC_df_train,y_train)
y_lr=classifier.predict(X_test_pc)

print('Confusion Matrix \n',confusion_matrix(y_lr,y_test))
print()
print('Accuracy Score \n', accuracy_score(y_lr,y_test))
print()
print('Classification Report \n',classification_report(y_lr,y_test))



#Random Forest Classifier - Using Grid Search to determine the best parameter model


#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
# Create the parameter grid based on the results of random search 

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
{'bootstrap': [True, False],
 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 'max_features': ['auto', 'sqrt'],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_pc,y_train)

rf_random.best_params_


#Results to be used to narrow down gridsearch model to continue parameter tuning
#{'n_estimators': 1000,
# 'min_samples_split': 2,
#'min_samples_leaf': 1,
#'max_features': 'auto',
# 'max_depth': 50,
#'bootstrap': False}



# Create a base model
rf = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

# Instantiate the grid search model Please note takes alot of time, paramaters are from Randomized search above

param_grid = {'bootstrap': [True, False],
              'max_depth': [10, 20, 30, 40, 50],
              'max_features': ['auto', 'sqrt'],
              'min_samples_leaf': [1],
              'min_samples_split': [2],
              'n_estimators': [200, 400, 600, 800, 1000]}


grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


grid_search.fit(X_train_pc,y_train)

grid_search.best_params_

# Results from Random Forest GridSearchCV 
#{'bootstrap': False,
# 'max_depth': 50,
# 'max_features': 'sqrt',
# 'min_samples_leaf': 1,
# 'min_samples_split': 2,
# 'n_estimators': 200}


#best_grid = grid_search.best_estimator_
#grid_accuracy = evaluate(best_grid, X_train_pc,y_train)




#Random Forest model without parameters
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_pc,y_train)
y_rfc=classifier.predict(X_test_pc)

print('Confusion Matrix \n',confusion_matrix(y_rfc,y_test))
print()
print('Accuracy Score \n', accuracy_score(y_rfc,y_test))
print()
print('Classification Report \n',classification_report(y_rfc,y_test))



#Random Forest model with GridSearch Parameters
classifier = RandomForestClassifier(bootstrap=False,max_depth=50,max_features='sqrt',min_samples_leaf=1,min_samples_split=2,n_estimators=200, random_state=42)
classifier.fit(X_train_pc,y_train)
y_rfc_GS=classifier.predict(X_test_pc)

print('Confusion Matrix \n',confusion_matrix(y_rfc_GS,y_test))
print()
print('Accuracy Score \n', accuracy_score(y_rfc_GS,y_test))
print()
print('Classification Report \n',classification_report(y_rfc_GS,y_test))



#Results Comparison Table


lr_df = pd.DataFrame(data=[f1_score(y_test,y_lr),accuracy_score(y_test, y_lr), recall_score(y_test, y_lr), precision_score(y_test, y_lr), roc_auc_score(y_test, y_lr)], 
             columns=['Logistic Regression Without Grid Search Parameters'], index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])

lr_df_GS = pd.DataFrame(data=[f1_score(y_test,y_lr_GS),accuracy_score(y_test,y_lr_GS), recall_score(y_test, y_lr_GS), precision_score(y_test, y_lr_GS), roc_auc_score(y_test, y_lr_GS)], 
             columns=['Logistic Regression using Grid Search Parameters'], index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])

rf_df = pd.DataFrame(data=[f1_score(y_test,y_rfc),accuracy_score(y_test, y_rfc), recall_score(y_test, y_rfc),precision_score(y_test, y_rfc), roc_auc_score(y_test, y_rfc)], 
             columns=['Random Forest Score '],index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])

rf_df_GS = pd.DataFrame(data=[f1_score(y_test,y_rfc_GS),accuracy_score(y_test, y_rfc_GS), recall_score(y_test, y_rfc_GS),precision_score(y_test, y_rfc_GS), roc_auc_score(y_test, y_rfc_GS)], 
             columns=['Random Forest Score with GridSearch Parameters '],index=["F1","Accuracy", "Recall", "Precision", "ROC AUC Score"])

df_models = round(pd.concat([lr_df,lr_df_GS,rf_df,rf_df_GS], axis=1),3)
colors = ["bisque","ivory","sandybrown","steelblue","lightsalmon"]
colormap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

background_color = "white"

fig = plt.figure(figsize=(18,26)) # create figure
gs = fig.add_gridspec(4, 2)
gs.update(wspace=0.1, hspace=0.5)
ax0 = fig.add_subplot(gs[0, :])

sns.heatmap(df_models.T, cmap=colormap,annot=True,fmt=".1%",vmin=0,vmax=0.95, linewidths=2.5,cbar=False,ax=ax0,annot_kws={"fontsize":16})
fig.patch.set_facecolor(background_color) # figure background color
ax0.set_facecolor(background_color) 

ax0.text(0,-0.5,'Model Comparison',fontsize=20,fontweight='bold',fontfamily='serif')
plt.show()




