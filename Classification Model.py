#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Student Name : Sheethal Melnarse
# Cohort       : 3 (Castro)

################################################################################
# Import Packages
################################################################################

import numpy as np
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # linear regression (statsmodels)
import sklearn.model_selection 
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.preprocessing import StandardScaler     # standard scaler
from sklearn.externals.six import StringIO           # saves objects in memory
from IPython.display import Image                    # displays on frontend
from sklearn.ensemble import GradientBoostingClassifier # gbm
from sklearn.model_selection import GridSearchCV     # hyperparameter tuning
from sklearn.metrics import make_scorer              # customizable scorer
import random as rand # random number generation

################################################################################
# Load Data
################################################################################

file = 'C:/Users/SHEETHAL/Downloads/Apprentice_Chef_Dataset.xlsx'
original_df = pd.read_excel(file)

# Renaming the file name

my_chef = original_df


################################################################################
# Feature Engineering 
################################################################################

# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in my_chef.iterrows():
    
    # splitting email domain at '@'
    split_email = my_chef.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)
    

# converting placeholder_lst into a DataFrame 
email_df = pd.DataFrame(placeholder_lst)

# STEP 2: concatenating with original DataFrame

# safety measure in case of multiple concatenations
original_df = pd.read_excel(file)

my_chef = original_df

# renaming column to concatenate
email_df.columns = ['NAME' , 'EMAIL_DOMAIN']


# concatenating personal_email_domain with friends DataFrame
my_chef = pd.concat([my_chef, email_df['EMAIL_DOMAIN']],
                   axis = 1)

# printing value counts of personal_email_domain
my_chef.loc[: ,'EMAIL_DOMAIN'].value_counts()

# email domain types
personal_domains = ['@gmail.com', '@protonmail.com', '@yahoo.com']
professional_domains  = ['@mmm.com','@amex.com','@apple.com','@boeing.com','@caterpillar.com','@chevron.com','@cisco.com','@cocacola.com','@disney.com','@dupont.com','@exxon.com','@ge.org','@goldmansacs.com','@homedepot.com','@ibm.com','@intel.com','@jnj.com','@jpmorgan.com','@mcdonalds.com','@merck.com','@microsoft.com','@nike.com','@pfizer.com','@pg.com','@travelers.com','@unitedtech.com','@unitedhealth.com','@verizon.com','@visa.com','@walmart.com']
junk_domains = ['@me.com','@aol.com','@hotmail.com','@live.com','@msn.com','@passport.com']

# placeholder list
placeholder_lst = []


# looping to group observations by domain type
for domain in my_chef['EMAIL_DOMAIN']:
    if '@' + domain in personal_domains:
        placeholder_lst.append('personal')
        
    elif '@' + domain in professional_domains:
        placeholder_lst.append('professional')
    
    elif '@' + domain in junk_domains:
        placeholder_lst.append('junk')
        
    else:
        print('Unknown')


# concatenating with original DataFrame
my_chef['DOMAIN_GROUP'] = pd.Series(placeholder_lst)

# Imputing the NULL values with 'Unknown'
my_chef['FAMILY_NAME'].fillna(value='Unknown')
my_chef = my_chef.drop('NAME', axis = 1)
my_chef = my_chef.drop('EMAIL', axis = 1)
my_chef = my_chef.drop('FIRST_NAME', axis = 1)
my_chef = my_chef.drop('FAMILY_NAME', axis = 1)

# Adding new features based on existing features
my_chef['REVENUE_PER_MEAL'] = my_chef['REVENUE'] / my_chef['TOTAL_MEALS_ORDERED']
my_chef['TOTAL_LOGINS'] = my_chef['PC_LOGINS'] + my_chef['MOBILE_LOGINS']
my_chef['AVG_TIME_PER_CLICK'] = my_chef['AVG_TIME_PER_SITE_VISIT']/my_chef['AVG_CLICKS_PER_VISIT']
my_chef['AVG_PHOTOS_VIEWED_PER_LOGIN'] = round(my_chef['TOTAL_PHOTOS_VIEWED']/my_chef['TOTAL_LOGINS'],2)


# Outlier threshold

TOTAL_MEALS_ORDERED_HI = 120
UNIQUE_MEALS_PURCH_HI = 7
CONTACTS_W_CUSTOMER_SERVICE_HI = 9
PRODUCT_CATEGORIES_VIEWED_HI = 6
AVG_TIME_PER_SITE_VISIT_HI = 160
CANCELLATIONS_BEFORE_NOON_HI = 4
CANCELLATIONS_BEFORE_NOON_LO = 0
CANCELLATIONS_AFTER_NOON_HI = 2
AVG_PREP_VID_TIME_HI = 230
FOLLOWED_RECOMMENDATIONS_PCT_HI = 40
AVG_CLICKS_PER_VISIT_HI = 16
TOTAL_PHOTOS_VIEWED_LO = 60
LARGEST_ORDER_SIZE_HI = 6
WEEKLY_PLAN_HI = 19
WEEKLY_PLAN_LO = 0
EARLY_DELIVERIES_HI = 5
EARLY_DELIVERIES_LO = 1
LATE_DELIVERIES_HI = 7
MASTER_CLASSES_ATTENDED_HI = 1.5
REVENUE_HI = 2100
REVENUE_PER_MEAL_HI = 50
AVG_TIME_PER_CLICK_HI = 12
AVG_PHOTOS_VIEWED_PER_LOGIN_AT = 0
AVG_PHOTOS_VIEWED_PER_LOGIN_HI = 45


# developing features (columns) for outliers

# TOTAL_MEALS_ORDERED
my_chef['OUT_TOTAL_MEALS_ORDERED'] = 0
condition_hi = my_chef.loc[0:,'OUT_TOTAL_MEALS_ORDERED'][my_chef['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

my_chef['OUT_TOTAL_MEALS_ORDERED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# UNIQUE_MEALS_PURCH
my_chef['OUT_UNIQUE_MEALS_PURCH'] = 0
condition_hi = my_chef.loc[0:,'OUT_UNIQUE_MEALS_PURCH'][my_chef['UNIQUE_MEALS_PURCH'] > UNIQUE_MEALS_PURCH_HI]

my_chef['OUT_UNIQUE_MEALS_PURCH'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CONTACTS_W_CUSTOMER_SERVICE
my_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'] = 0
condition_hi = my_chef.loc[0:,'OUT_CONTACTS_W_CUSTOMER_SERVICE'][my_chef['CONTACTS_W_CUSTOMER_SERVICE'] > CONTACTS_W_CUSTOMER_SERVICE_HI]

my_chef['OUT_CONTACTS_W_CUSTOMER_SERVICE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# PRODUCT_CATEGORIES_VIEWED
my_chef['OUT_PRODUCT_CATEGORIES_VIEWED'] = 0
condition_hi = my_chef.loc[0:,'OUT_PRODUCT_CATEGORIES_VIEWED'][my_chef['PRODUCT_CATEGORIES_VIEWED'] > PRODUCT_CATEGORIES_VIEWED_HI]

my_chef['OUT_PRODUCT_CATEGORIES_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_TIME_PER_SITE_VISIT
my_chef['OUT_AVG_TIME_PER_SITE_VISIT'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_TIME_PER_SITE_VISIT'][my_chef['AVG_TIME_PER_SITE_VISIT'] > AVG_TIME_PER_SITE_VISIT_HI]

my_chef['OUT_AVG_TIME_PER_SITE_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_BEFORE_NOON
my_chef['OUT_CANCELLATIONS_BEFORE_NOON'] = 0
condition_hi = my_chef.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][my_chef['CANCELLATIONS_BEFORE_NOON'] > CANCELLATIONS_BEFORE_NOON_HI]
condition_lo = my_chef.loc[0:,'OUT_CANCELLATIONS_BEFORE_NOON'][my_chef['CANCELLATIONS_BEFORE_NOON'] < CANCELLATIONS_BEFORE_NOON_LO]

my_chef['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

my_chef['OUT_CANCELLATIONS_BEFORE_NOON'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# CANCELLATIONS_AFTER_NOON
my_chef['OUT_CANCELLATIONS_AFTER_NOON'] = 0
condition_hi = my_chef.loc[0:,'OUT_CANCELLATIONS_AFTER_NOON'][my_chef['CANCELLATIONS_AFTER_NOON'] > CANCELLATIONS_AFTER_NOON_HI]

my_chef['OUT_CANCELLATIONS_AFTER_NOON'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_PREP_VID_TIME
my_chef['OUT_AVG_PREP_VID_TIME'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_PREP_VID_TIME'][my_chef['AVG_PREP_VID_TIME'] > AVG_PREP_VID_TIME_HI]

my_chef['OUT_AVG_PREP_VID_TIME'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# FOLLOWED_RECOMMENDATIONS_PCT
my_chef['OUT_FOLLOWED_RECOMMENDATIONS_PCT'] = 0
condition_hi = my_chef.loc[0:,'OUT_FOLLOWED_RECOMMENDATIONS_PCT'][my_chef['FOLLOWED_RECOMMENDATIONS_PCT'] > FOLLOWED_RECOMMENDATIONS_PCT_HI]

my_chef['OUT_FOLLOWED_RECOMMENDATIONS_PCT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# EARLY_DELIVERIES
my_chef['OUT_EARLY_DELIVERIES'] = 0
condition_hi = my_chef.loc[0:,'OUT_EARLY_DELIVERIES'][my_chef['EARLY_DELIVERIES'] > EARLY_DELIVERIES_HI]
condition_lo = my_chef.loc[0:,'OUT_EARLY_DELIVERIES'][my_chef['EARLY_DELIVERIES'] < EARLY_DELIVERIES_LO]

my_chef['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

my_chef['OUT_EARLY_DELIVERIES'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# AVG_CLICKS_PER_VISIT
my_chef['OUT_AVG_CLICKS_PER_VISIT'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_CLICKS_PER_VISIT'][my_chef['AVG_CLICKS_PER_VISIT'] > AVG_CLICKS_PER_VISIT_HI]

my_chef['OUT_AVG_CLICKS_PER_VISIT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# TOTAL_PHOTOS_VIEWED
my_chef['OUT_TOTAL_PHOTOS_VIEWED'] = 0
condition_hi = my_chef.loc[0:,'OUT_TOTAL_PHOTOS_VIEWED'][my_chef['TOTAL_PHOTOS_VIEWED'] < TOTAL_PHOTOS_VIEWED_LO]

my_chef['OUT_TOTAL_PHOTOS_VIEWED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# LARGEST_ORDER_SIZE
my_chef['OUT_LARGEST_ORDER_SIZE'] = 0
condition_hi = my_chef.loc[0:,'OUT_LARGEST_ORDER_SIZE'][my_chef['LARGEST_ORDER_SIZE'] > LARGEST_ORDER_SIZE_HI]

my_chef['OUT_LARGEST_ORDER_SIZE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# WEEKLY_PLAN
my_chef['OUT_WEEKLY_PLAN'] = 0
condition_hi = my_chef.loc[0:,'OUT_WEEKLY_PLAN'][my_chef['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]
condition_lo = my_chef.loc[0:,'OUT_WEEKLY_PLAN'][my_chef['WEEKLY_PLAN'] < WEEKLY_PLAN_LO]

my_chef['OUT_WEEKLY_PLAN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

my_chef['OUT_WEEKLY_PLAN'].replace(to_replace = condition_lo,
                                value      = 1,
                                inplace    = True)

# LATE_DELIVERIES
my_chef['OUT_LATE_DELIVERIES'] = 0
condition_hi = my_chef.loc[0:,'OUT_LATE_DELIVERIES'][my_chef['LATE_DELIVERIES'] > LATE_DELIVERIES_HI]

my_chef['OUT_LATE_DELIVERIES'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# MASTER_CLASSES_ATTENDED
my_chef['OUT_MASTER_CLASSES_ATTENDED'] = 0
condition_hi = my_chef.loc[0:,'OUT_MASTER_CLASSES_ATTENDED'][my_chef['MASTER_CLASSES_ATTENDED'] > MASTER_CLASSES_ATTENDED_HI]

my_chef['OUT_MASTER_CLASSES_ATTENDED'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)


# REVENUE
my_chef['OUT_REVENUE'] = 0
condition_hi = my_chef.loc[0:,'OUT_REVENUE'][my_chef['REVENUE'] > REVENUE_HI]

my_chef['OUT_REVENUE'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# REVENUE_PER_MEAL
my_chef['OUT_REVENUE_PER_MEAL'] = 0
condition_hi = my_chef.loc[0:,'OUT_REVENUE_PER_MEAL'][my_chef['REVENUE_PER_MEAL'] > REVENUE_PER_MEAL_HI]

my_chef['OUT_REVENUE_PER_MEAL'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_TIME_PER_CLICK
my_chef['OUT_AVG_TIME_PER_CLICK'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_TIME_PER_CLICK'][my_chef['AVG_TIME_PER_CLICK'] > AVG_TIME_PER_CLICK_HI]

my_chef['OUT_AVG_TIME_PER_CLICK'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

# AVG_PHOTOS_VIEWED_PER_LOGIN
my_chef['OUT_AVG_PHOTOS_VIEWED_PER_LOGIN'] = 0
condition_hi = my_chef.loc[0:,'OUT_AVG_PHOTOS_VIEWED_PER_LOGIN'][my_chef['AVG_PHOTOS_VIEWED_PER_LOGIN'] > AVG_PHOTOS_VIEWED_PER_LOGIN_HI]
condition_at = my_chef.loc[0:,'OUT_AVG_PHOTOS_VIEWED_PER_LOGIN'][my_chef['AVG_PHOTOS_VIEWED_PER_LOGIN'] == AVG_PHOTOS_VIEWED_PER_LOGIN_AT]

my_chef['OUT_AVG_PHOTOS_VIEWED_PER_LOGIN'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)
my_chef['OUT_AVG_PHOTOS_VIEWED_PER_LOGIN'].replace(to_replace = condition_at,
                                value      = 1,
                                inplace    = True)

################################################################################
# One hot encoding Median Meal Rating and Domain Group
################################################################################

# one hot encoding categorical variables
ONE_HOT_MEDIAN_MEAL_RATING_1 = pd.get_dummies(my_chef['MEDIAN_MEAL_RATING'])
ONE_HOT_MEDIAN_MEAL_RATING_2 = pd.get_dummies(my_chef['DOMAIN_GROUP'])

# dropping categorical variables after they've been encoded
my_chef = my_chef.drop('MEDIAN_MEAL_RATING', axis = 1)
my_chef = my_chef.drop('DOMAIN_GROUP', axis = 1)

# joining codings together
my_chef = my_chef.join([ONE_HOT_MEDIAN_MEAL_RATING_1])
my_chef = my_chef.join([ONE_HOT_MEDIAN_MEAL_RATING_2])



# Renaming one hot encoded columns
my_chef.columns = ['REVENUE','CROSS_SELL_SUCCESS','TOTAL_MEALS_ORDERED','UNIQUE_MEALS_PURCH','CONTACTS_W_CUSTOMER_SERVICE',
                  'PRODUCT_CATEGORIES_VIEWED','AVG_TIME_PER_SITE_VISIT','MOBILE_NUMBER','CANCELLATIONS_BEFORE_NOON',
                  'CANCELLATIONS_AFTER_NOON','TASTES_AND_PREFERENCES','PC_LOGINS','MOBILE_LOGINS','WEEKLY_PLAN','EARLY_DELIVERIES',
                  'LATE_DELIVERIES','PACKAGE_LOCKER','REFRIGERATED_LOCKER','FOLLOWED_RECOMMENDATIONS_PCT','AVG_PREP_VID_TIME',
                  'LARGEST_ORDER_SIZE','MASTER_CLASSES_ATTENDED','AVG_CLICKS_PER_VISIT','TOTAL_PHOTOS_VIEWED','EMAIL_DOMAIN',
                  'REVENUE_PER_MEAL','TOTAL_LOGINS','AVG_TIME_PER_CLICK','AVG_PHOTOS_VIEWED_PER_LOGIN',
                  'OUT_TOTAL_MEALS_ORDERED','OUT_UNIQUE_MEALS_PURCH','OUT_CONTACTS_W_CUSTOMER_SERVICE','OUT_PRODUCT_CATEGORIES_VIEWED',
                  'OUT_AVG_TIME_PER_SITE_VISIT','OUT_CANCELLATIONS_BEFORE_NOON','OUT_CANCELLATIONS_AFTER_NOON','OUT_AVG_PREP_VID_TIME',
                  'OUT_FOLLOWED_RECOMMENDATIONS_PCT','OUT_EARLY_DELIVERIES','OUT_AVG_CLICKS_PER_VISIT','OUT_TOTAL_PHOTOS_VIEWED',
                  'OUT_LARGEST_ORDER_SIZE','OUT_WEEKLY_PLAN','OUT_LATE_DELIVERIES','OUT_MASTER_CLASSES_ATTENDED','OUT_REVENUE',
                  'OUT_REVENUE_PER_MEAL','OUT_AVG_TIME_PER_CLICK','OUT_AVG_PHOTOS_VIEWED_PER_LOGIN','MEDIAN_MEAL_RATING_1',
                  'MEDIAN_MEAL_RATING_2','MEDIAN_MEAL_RATING_3','MEDIAN_MEAL_RATING_4','MEDIAN_MEAL_RATING_5', 'JUNK', 'PERSONAL', 'PROFESSIONAL']

# saving feature-rich dataset in Excel
my_chef.to_excel('my_chef_feature_classification.xlsx',
                 index = False)


################################################################################
# Train/Test Split
################################################################################

# creating a dictionary to store features

Feature_dict = {
    
 # significant variables only
 'sig_var'    : ['MOBILE_NUMBER' , 'CANCELLATIONS_BEFORE_NOON', 'TASTES_AND_PREFERENCES', 'MOBILE_LOGINS',
                   'OUT_AVG_CLICKS_PER_VISIT','OUT_TOTAL_PHOTOS_VIEWED', 'PROFESSIONAL','JUNK','OUT_AVG_PHOTOS_VIEWED_PER_LOGIN']
}

# declaring explanatory variables
my_chef_data   = my_chef.loc[ : , Feature_dict['sig_var']]

# preparing response variable
my_chef_target = my_chef.loc[:,'CROSS_SELL_SUCCESS']

# INSTANTIATING StandardScaler()
scaler = StandardScaler()

# FITTING the data
scaler.fit(my_chef_data)

# TRANSFORMING the data
X_scaled     = scaler.transform(my_chef_data)

# converting to a DataFrame
X_scaled_df  = pd.DataFrame(X_scaled) 

# train-test split with the scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
            X_scaled_df,
            my_chef_target,
            random_state = 222,
            test_size = 0.25,
            stratify = my_chef_target)


################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING the model object with hyperparameters
gbm_classifier = GradientBoostingClassifier(loss = 'deviance',
                                              learning_rate = 0.3,
                                              n_estimators  = 100,
                                              criterion     = 'friedman_mse',
                                              max_depth     = 3,
                                              warm_start    = False,
                                              random_state  = 222)


# FIT step is needed as we are not using .best_estimator
gbm_classifier_fit = gbm_classifier.fit(X_train_scaled, y_train_scaled)


# PREDICTING based on the testing set
gbm_classifier_pred = gbm_classifier_fit.predict(X_test_scaled)



################################################################################
# Final Model Score (score)
################################################################################

print('test_score:', gbm_classifier_fit.score(X_test_scaled, y_test_scaled).round(3))
print('AUC Score :', roc_auc_score(y_true  = y_test_scaled,
                                          y_score = gbm_classifier_pred).round(3))


# In[ ]:




