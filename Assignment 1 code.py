# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 15:57:06 2021

@author: Ronal
"""

# Data manipulation
import pandas as pd
import numpy as np

# Data visualization
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# preprocessing
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer


# Ml model just putting in code now as I may want to use for part 2 etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


# General Libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from collections import Counter
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")



##Importing Bankruptcy Data CSV code
#importing data as a dataframe
filename = 'data.csv'
Bankruptcy_Data = pd.read_csv(filename)
Bankruptcy_Data.head()
Bankruptcy_Data.info()


      
## Checking Nan presence

[print(col) for col in Bankruptcy_Data if Bankruptcy_Data[col].isna().sum() > 0]

## Checking for duplicates

Bankruptcy_Data.duplicated().sum()

###Summary Stats of Data
Bankruptcy_Summary_stats=Bankruptcy_Data.describe()
print(Bankruptcy_Summary_stats)


#how many rows and columns in dataset
Bankruptcy_Data.shape
#6819 rows, 96 columns


Total_Bankruptcies = Bankruptcy_Data['Bankrupt?'].sum()
print(Total_Bankruptcies)
#there are 220 bankruptcies in the dataset


###Code fount for portray distribution of bankruptcies alternatively
# The classes are heavily skewed we need to solve this issue later.

print(Bankruptcy_Data['Bankrupt?'].value_counts())
print('-'* 30)
print('Financially stable/Non-Bankrupt: ', round(Bankruptcy_Data['Bankrupt?'].value_counts()[0]/len(Bankruptcy_Data) * 100,2), '% of the dataset')
print('Financially unstable/Bankrupt: ', round(Bankruptcy_Data['Bankrupt?'].value_counts()[1]/len(Bankruptcy_Data) * 100,2), '% of the dataset')

######
##Checking Lable Distribution

sns.set_theme(context = 'paper')

plt.figure(figsize = (10,5))
sns.countplot(Bankruptcy_Data['Bankrupt?'])
plt.title('Class Distributions \n (0: Fin. Non-Bankrupt || 1: Fin. Bankrupt)', fontsize=14)
plt.show()

# Plotting interesting features
f, axes = plt.subplots(ncols=4, figsize=(24,6))

sns.boxplot(x='Bankrupt?', y=" Net Income to Total Assets", data=Bankruptcy_Data, ax=axes[0])
axes[0].set_title('Bankrupt vs Net Income to Total Assets')

sns.boxplot(x='Bankrupt?', y=" Total debt/Total net worth", data=Bankruptcy_Data, ax=axes[1]) 
axes[1].set_title('Bankrupt vs Tot Debt/Net worth Correlation')


sns.boxplot(x='Bankrupt?', y=" Debt ratio %", data=Bankruptcy_Data, ax=axes[2])
axes[2].set_title('Bankrupt vs Debt ratio Correlation')


sns.boxplot(x='Bankrupt?', y=" Net worth/Assets", data=Bankruptcy_Data, ax=axes[3])  
axes[3].set_title('Bankrupt vs Net Worth/Assets Correlation') 

plt.show()


# Plotting other interesting features

f, axes = plt.subplots(ncols=4, figsize=(24,6))

sns.boxplot(x='Bankrupt?', y=" Working Capital to Total Assets", data=Bankruptcy_Data, ax=axes[0])
axes[0].set_title('Bankrupt vs  working capital to total assets')

sns.boxplot(x='Bankrupt?', y=" Cash/Total Assets", data=Bankruptcy_Data, ax=axes[1])
axes[1].set_title('Bankrupt vs cash / total assets')


sns.boxplot(x='Bankrupt?', y=" Current Liability to Assets", data=Bankruptcy_Data, ax=axes[2])
axes[2].set_title('Bankrupt vs current liability to assets')


sns.boxplot(x='Bankrupt?', y=" Retained Earnings to Total Assets", data=Bankruptcy_Data, ax=axes[3])
axes[3].set_title('Bankrupt vs  Retained Earnings to Total Assets')

plt.show()


## Code used for Appendices
# Plotting the feature distributions for close to bankrputcy companies

f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(24, 6))

cash_flow_rate = Bankruptcy_Data[' Net Income to Total Assets'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(cash_flow_rate,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title(' Net Income to Total Assets \n (Unstable companies)', fontsize=14)

tot_debt_net = Bankruptcy_Data[' Total debt/Total net worth'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(tot_debt_net ,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('total debt/tot net worth \n (Unstable companies)', fontsize=14)


debt_ratio = Bankruptcy_Data[' Debt ratio %'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(debt_ratio,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('debt_ratio \n (Unstable companies)', fontsize=14)

net_worth_assets = Bankruptcy_Data[' Net worth/Assets'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(net_worth_assets,ax=ax4, fit=norm, color='#C5B3F9')
ax4.set_title('net worth/assets \n (Unstable companies)', fontsize=14)

plt.show()




f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(24, 6))

working_cap = Bankruptcy_Data[' Working Capital to Total Assets'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(working_cap,ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('working capitals to total assets \n (Unstable companies)', fontsize=14)

cash_tot_assets = Bankruptcy_Data[' Cash/Total Assets'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(cash_tot_assets ,ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('cash/total assets \n (Unstable companies)', fontsize=14)


asset_liab = Bankruptcy_Data[' Current Liability to Assets'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(asset_liab,ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('liability to assets \n (Unstable companies)', fontsize=14)

operating_funds = Bankruptcy_Data[' Retained Earnings to Total Assets'].loc[Bankruptcy_Data['Bankrupt?'] == 1].values
sns.distplot(operating_funds,ax=ax4, fit=norm, color='#C5B3F9')
ax4.set_title('retain earnings to total assets \n (Unstable companies)', fontsize=14)

plt.show()



#### code below is just for practice/playing around
#Subsetting Dataframe for Bankrupt entities and non-bankrupt entities
#Bankrupt subset
Y_Bankrupt_data=Bankruptcy_Data[Bankruptcy_Data["Bankrupt?"]==1]

YBankrupt_SumStat=Y_Bankrupt_data.describe()

#Non Bankrupt subset
Non_Bankrupt_data=Bankruptcy_Data[Bankruptcy_Data["Bankrupt?"]==0]

NonBankrupt_SumStat=Non_Bankrupt_data.describe()

#### Code to calculate the mean of each column df.mean(axis=0), axis=0 argument calculates the column wise, axis=1 row wise
#Non Bankrupt subset Mean
Non_Bankrupt_data_col_Mean=Non_Bankrupt_data.mean(axis=0)
Non_Bankrupt_data_col_Mean
#Bankrupt subset Mean
Bankrupt_data_col_Mean=Y_Bankrupt_data.mean(axis=0)
Bankrupt_data_col_Mean

#Combining the mean subsets for analysis
#df = pd. concat([a_series, another_series], axis=1) merge `a_series` and `another_series`

B_NB_Data_Combined=pd.concat([Non_Bankrupt_data_col_Mean, Bankrupt_data_col_Mean], axis=1)

#changing column names to make it more readable .columns example
B_NB_DataFrame=pd.DataFrame(B_NB_Data_Combined)

#Creating visualisations for Part 1 of the report 







