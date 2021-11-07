#!/usr/bin/env python
# coding: utf-8

# <img src = "https://github.com/barcelonagse-datascience/academic_files/raw/master/bgsedsc_0.jpg">
# $\newcommand{\bb}{\boldsymbol{\beta}}$
# $\DeclareMathOperator{\Gau}{\mathcal{N}}$
# $\newcommand{\bphi}{\boldsymbol \phi}$
# $\newcommand{\bx}{\boldsymbol{x}}$
# $\newcommand{\bu}{\boldsymbol{u}}$
# $\newcommand{\by}{\boldsymbol{y}}$
# $\newcommand{\whbb}{\widehat{\bb}}$
# $\newcommand{\hf}{\hat{f}}$
# $\newcommand{\tf}{\tilde{f}}$
# $\newcommand{\ybar}{\overline{y}}$
# $\newcommand{\E}{\mathbb{E}}$
# $\newcommand{\Var}{Var}$
# $\newcommand{\Cov}{Cov}$
# $\newcommand{\Cor}{Cor}$

# # Project: Linear models

# ## Programming project: real estate assesment evaluation
# 
# Home valuation is key in real estate industry, and also the basis for mortgages in credit sector. Here we have to predict the estimated value of a property.
# 
# 
# Data (*Regression_Supervised_Train.csv*) consist of a list of features plus the resulting <i>parcelvalue</i>, described in *Case_data_dictionary.xlsx* file. Each row corresponds to a particular home valuation, and <i>transactiondate</i> is the date when the property was effectively sold. Properties are defined by <i>lotid</i>, but be aware that one property can be sold more than once (it's not the usual case). Also notice that some features are sometime empty, your model has to deal with it.
# 
# Note that you shouldn't use <i>totaltaxvalue</i>, <i>buildvalue</i> or <i>landvalue</i>, because they are closely correlated with the final value to predict. There is a further member of the training set predictors which is not available in the test set and therefore needs removing. 
# 
# + Using this data build a predictive model for <i>parcelvalue</i> 
# + In your analysis for faster algorithms use the AIC criterion for choosing any hyperparameters 
# + Try a first quick implementation, then try to optimize hyperparameters
# + For this analysis there is an extra test dataset. Once your code is submitted we will run a competition to see how you score in the test data. Hence have prepared also the necessary script to compute the MSE estimate on the test data once released.
# + Bonus: Try an approach to fill NA without removing features or observations, and check improvements.
# 
# You can follow those **steps** in your first implementation:
# 1. *Explore* and understand the dataset. Report missing data
# 2. As a simplified initial version, get rid of *missing data* by:
#     + Remove columns '<i>totaltaxvalue</i>', '<i>buildvalue</i>' or '<i>landvalue</i>' from the training and testing set and also '<i>mypointer</i>' from the training set
#     + Removing features that have more than 40% of missing data in the training set (remember anything you remove from the training set must be removed form the testing set!) (HINT: data.dropna(axis=1, thresh=round(my_percentage_valid*len(data.index)) - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html)
#     + After that, removing observations that have missing data
# 3. Create *dummy variables* for relevant categorical features (EXTENDED PROJECT ONLY)
# 4. *Build* your model and test it on the same input data
# 5. Assess expected accuracy using *cross-validation*
# 6. Report which variable impacts more on results 
# 7. Prepare the code to *run* on a new input file and be able to report accuracy, following same preparation steps (missing data, dummies, etc)
# 
# You may want to iterate to refine some of these steps once you get performance results in step 5.
# 

# ## Main criteria for grading
# From more to less important (the weighting of these components will vary between the in-class and extended projects):
# + Code runs
# + Parcel value prediction made
# + Accuracy of predictions for test properties is calculated (kaggle)
# + Linear Models(s) and LASSO have been used
# + Accuracy itself
# + Data preparation
# + Hyperparameter optimization
# + Code is combined with neat and understandable commentary, with some titles and comments (demonstrate you have understood the methods and the outputs produced)
# + Improved methods from what we discussed in class (properly explained/justified)

# # My Linear Regression Project



from my_functions import *


# ### Data preparation



## Your code here (click on the window and type 'b' if you want to split in more than one code window)

# Step 1: Read data, report missing data

import pandas as pd #importing pandas
train = pd.read_csv('C:/Users/sssalas/OneDrive - Philippine Competition Commission/Desktop/project1/real-estate-valuation-with-linear-models/Regression_Supervised_Train.csv') 
#reading train dataset
test = pd.read_csv('C:/Users/sssalas/OneDrive - Philippine Competition Commission/Desktop/project1/real-estate-valuation-with-linear-models/Regression_Supervised_Test.csv')  
#reading test dataset

datasets = [train, test]

for i in datasets:
    print(i.shape)
# there are two more columns in the train dataset, 'parcelvalue' and 'mypointer'




# Make a copy of data for later use
train_original=train.copy()
test_original=test.copy()




to_drop = ['totaltaxvalue', 'buildvalue', 'landvalue', 'mypointer']


#dropping the columns suggested to be deleted in the instructions to avoid high correlation
train = train.drop(to_drop, axis=1)

test = test.drop(to_drop[:3], axis=1)
#because we dropped these value columns, it would be quite difficult to get a low mean square error



datasets = [train, test]

for i in datasets:
    print(i.shape)
# now, the two datasets only differ in features in terms of parcel value



# counting observations per feature in the train and test dataset with null values
for i in datasets:
    print(i.isnull().sum())



# Step 2: Remove features with missing data, and then observations with missing data
train = train.dropna(axis=1, thresh=round(0.60*len(train.index)))
#any feature with  more than 40% missing values will be removed. in other words, we keep only the columns whose 60% of the values are not NaNs

train.shape #checking the dimensions, now we only have 19 features as opposed to 44 before. the number of our observations is 24755.



train_columns = train.columns #we store the column names that have been left in the train dataset. this will be our index for the columns that we will retain for the test dataset
train_columns = train_columns.drop(["parcelvalue"])


test = test[train_columns]
# now, we only retain columns in the test dataset that are present in the train dataset as well


test.shape
# we confirm that there are only 18 features left in our test dataset. the train dataset still has more than 1 column (parcelvalue)


datasets = [train, test]

for i in datasets:
    print(i.isnull().sum())



display(train) #we notice that there are still observations in train that have missing values



train = train.dropna() #we drop observations that have missing values in the train dataset, 
#not necessary to put axis=0 because that is the default

train.shape #now the dimensions of our train dataset is (12560, 19). The number of our observations is 12560. This is about half of the number of observations before.


display(train)


test = test.dropna() #we drop observations that have missing values in the train dataset


test.shape #now the dimensions of our train dataset is (2746, 18).
# we still are left with the same number of obs and columns because there are no missing values in the rows of the train dataset


display(test)


datasets = [train, test]

for i in datasets:
    print(i.isnull().sum())

# now, we can confirm that there are no more null values in our datasets


#we notice that there are two countycodes, therefore we check if they are one and the same using corr. if there is perfect correlation, we drop one of them
corr_county = train['countycode'].corr(train['countycode2'])
print(corr_county) #corr_county = -1.0, therefore there is perfect correlation. 
#therefore we drop county2 in both datasets. note that this is only the purpose of this model. 
#it might be the case that the neighborhood codes hold other informations that are not just categorical (e.g. proximity)

#we also drop lotid because it has no cardinal or ordinal meaning 


# we have also told to dropped the lotid because this name is arbitararily set and wouldn't help us with our pracelvalue predictions

train = train.drop(['countycode2', 'lotid'], axis=1) #now we only have 17 columns in train dataset
test = test.drop(['countycode2', 'lotid'], axis=1) #now we only have 16 columns in test dataset


train.describe() #we check some summary statistics of our train dataset
#at the same time we get a clear view of the columns that are left


test.describe() #we check some summary statistics of our test dataset
#at the same time we get a clear view of the columns that are left


print(train.shape)
print(test.shape)
#parcelvalue is the only difference in column



y_train = train.loc[:,'parcelvalue'] #we isolate the dependent variables from the whole training dataset
print(y_train)


import seaborn as sns
#histogram

sns.displot(pd.Series(y_train))
#we see that the distribution of parcelvalue is highly positively skewed, so we might want to trasnform this to log
#this will change the interpretation of the coefficients
#but since we are doing machine learning, this step is not necessary



X_train = train.drop(['parcelvalue'], axis=1) #we create our the regressors training dataset by creating a dataframe that contains all elements of the train dataset execpt parcelvalue

X_train.head()


X_train.shape # dimensions (12560, 570), one less than the train dataset itself because we do not have 'parcelvalue' here


X_test = test



X_test.head()



X_test.shape


# In[184]:


print(X_train.shape)
print(X_test.shape)
#same dimensions as X_train


# ### Building the first model

# Note that I will be repeating steps 4 to 8 multiple times since I will have different models, transformations of features, and parameter optimizations.
# 
# 

# **Linear Regression using non-transformed features**
# - we first try to fit our model based on non-transformed dataset

# In[185]:


# Step 4: Build your model and get predictions from train data
from sklearn.linear_model import LinearRegression #import the Linear Regression 
regr = LinearRegression() #store the function to an object

regr.fit(X_train,y_train) #we fit the model
y_hat_train = regr.predict(X_train) #we produce predictions from our fitted model based on test data


# In[186]:


print(y_hat_train)
y_hat_train.shape


# In[187]:


# Step 5: Assess expected accuracy

#we first assess the expected accurary of the model we have fitted using the original train and test datasets (the one without polynomial features and dummies)
##in-sample
import matplotlib.pylab as plt
from sklearn.metrics import r2_score

plt.figure() #creating a blank plot
plt.scatter(x=y_train,y=y_hat_train) #plotting the points
plt.plot(y_train,y_train,c="red") #plotting a 45 degree line
r_squared = r2_score(y_train, y_hat_train) #i use r_2 score instead of what Jack uses in class (getting correlation then squaring) because this is cleaner and more straightforward
plt.title('R-squared equals %.3f' %r_squared) 

# the value of R-squared is 0.485, there is some correlation between the y_train but not strong
#note that the interpretation of the red line is this: if all points lie on the red line, then there is perfect correlation
# we see that this is true for lower values of y_train but not for higher ones
#we can say that higher values are not predicted well by our model.
#therefore later on we need to optimize paramaters


# In[188]:


#I keep on using this kind of plot, so I will create a function
# I have already imported this custom function in the beginning

#def my_r2_plot(y_train, y_hat_train):
    #plt.figure() 
    #plt.scatter(x=y_train,y=y_hat_train) 
    #plt.plot(y_train,y_train,c="red") 
    #r_squared = r2_score(y_train, y_hat_train)
    #plt.title('R-squared equals %.3f' %r_squared) 


# In[189]:


##producing cross-validated predictions
from sklearn.model_selection import cross_val_predict as cvp
y_hat_cv = cvp(regr, X_train, y_train, cv=80) # #try first cv=80 then leave-one-out CV when cv=12560      
                                  # and 12560 because n=12560 #leave-one-out taking so long so i'll put 50 first (100 k-folds)

#If a value for k is chosen that does not evenly split the data sample, 
#then one group will contain a remainder of the examples. 
#It is preferable to split the data sample into k groups with the same number of samples, 
#such that the sample of model skill scores are all equivalent. 
#from https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85
#I'll use 80 since it's a factor of 12560

my_r2_plot(y_train, y_hat_cv)

# the value of R-squared is 0.480, there is some correlation between the y_train but not strong
#therefore go back to previous steps and optimize paramaters

#also, note that the r-squared with cross-validated predictions is lower than the usual in-sample r-squared (0.480 < 0.485).
# in other words, the performance of the model decreased, but not drastically

#this is as expected since the fitted model will be better to produce predictions based on the data it has seen before than on the data it has never since
#but also note that the r-squares of the in-sample and out-sample are not that far. therefore we could say that there is no overfitting (the model is bad both on preicting in-sample and out-sample data)


# In[190]:


#we could also use cross_val_score to assess expect accuracy 

#from sklearn.model_selection import cross_val_score as cvs

#accuracy = cvs(regr, X_train, y_train, scoring='accuracy', cv = 10)
#print(accuracy)

##but for some reason, the above code is not working as expected,
#checking stack exchange, the code may not be working because of the version of python that I'm using


# In[191]:


#We produce predicted values using leave-one-out cv just to see if it would be very different from cv=80.
y_hat_cv_LOO = cvp(regr, X_train, y_train, cv=2746) #this thing is taking so long

my_r2_plot(y_train,y_hat_cv_LOO )


# In[192]:


print(r2_score(y_hat_cv,y_hat_cv_LOO))
#we just use cv=80 instead of leave-one-one since they have a large correlation (0.999937568174697) anyway


# In[193]:


#Step 6:
print(regr.coef_)


# In[194]:


#recall that
print(X_train.columns)


# In[195]:


dict_zip(X_train.columns,regr.coef_)


# So this means that:
# - 1 additional bathroom increases the parcel value by 1326.20
# - 1 additional bedroom decreass the parcel value by 129300.41
# - 1 additional unit of finished are increases the parcel value by 460.58
# - and so on and so forth. 
# 
# Some have counter intuitive signs but we will ignore this for now as this is not our final model.
# Further, the one with the highest magnitude is number of bedrooms, with an absolute value of 129,300, which is also counterintuitive. But we will ignore this for now.

# In[196]:


#Step 7: Prepare code to run and check performance of you model using a new input data with same exact format
y_hat_test = regr.predict(X_test) #we produce predictions from our fitted model based on test data


# ### Some more data preparation

# **Polynomial Features and Dummy Variables**

# Next we try the model with polynomial features and dummies.
# 
# I carefully separate the variables that I will use for plf and dummies.
# 
# I separated them because for me, it does not make much sense to generate polynomial features of dummy variables since this will just generate more duplicate columns (with terms such as 1^2, 1^3, 0^2, 0^3 whihch does not make any sense, and would just make perfectly collineary columns) and would make the computation time too long.
# 
# Note to self: careful to make the variable names not confusing

# Let f be a shorthand for feature (to mark without making the var names too long) and let d be a shorthand for dummies.
# 
# I thought of not getting the polynomial features of year and tax year because for me, it makes little sense to get variable transformations of these, therefore I will not include them in my feautures dataset

# *Dummy Variable Generation*

# Let's check first the number of unique variables that we plan to have dummies on.

# In[197]:


potential = ['heatingtype', 'citycode', 'countycode', 'neighborhoodcode', 'regioncode']


# In[198]:


for i in potential:
    print(len(pd.unique(train[i])))

for i in potential:
    print(len(pd.unique(test[i])))


# In[246]:


def len_unique_print(c,d):
    for i in d:
        print(len(pd.unique(c[i])))


# In[247]:


len_unique_print(train,potential)


# We observe that there are too many unique values for neighborhoodcode and regioncode for both the train and test datasets
# We will also create dummy for regioncode and neighborhoodcode but we will impose additional constraints on which values we are making dummies on. I only want to create dummies for neighborhoods and regions that occurs frequently in our dataset. Hence, I will use a cut-off.
# 
# We prepare our neigborhoodcode and regioncode columns so that they won't be too many when we generate dummies. I will do the one for neighborhoods first.

# Later on, it is also important to make sure that the dummies in the train dataset will be the same as in the test dataset. 

# In[200]:


neighborhood_stats = X_train['neighborhoodcode'].value_counts(ascending=False) #we count the number of unique values, sort them from highest to lowest, store the results to neighborhood_stats
neighborhood_stats


# In[201]:


neighborhood_stats.values.sum() #checking if it would sum to total number of obs, and it is


# In[202]:


len(neighborhood_stats[neighborhood_stats>100]) #counting neighborhoodcodes with more than 100 obs
#I've decided that I am okay with have 30+1-1 dummies (neighborhoodcodes with more than 100 obs + others - drop_first)


# In[203]:


len(neighborhood_stats[neighborhood_stats<=100]) #there are 307 neighborhoods with less than 100 obs
#all of these neighborhoods will be identified as others later


# In[204]:


neighborhood_stats_less_than_100 = neighborhood_stats[neighborhood_stats<=100] #we store the neighborhoods will less than 100 obs
neighborhood_stats_less_than_100


# In[205]:


neighborhood_stats_less_than_100.shape #contains one column of the number of obs of neighborhoods with less than 100 obs


# In[206]:


len(X_train.neighborhoodcode.unique())


# In[207]:


X_train.neighborhoodcode = X_train.neighborhoodcode.apply(lambda x: 'other' if x in neighborhood_stats_less_than_100 else x) #we are replacing the values with 'other' for those neighborhoods with less than 100 obs
len(X_train.neighborhoodcode.unique()) #now, we only have 31 unique neighborhoodcodes instead of 337


# In[208]:


X_train.head()


# Now, doing the same for regions.

# In[209]:


region_stats = X_train['regioncode'].value_counts(ascending=False) #we count the number of unique values, sort them from highest to lowest, store the results to region_stats
region_stats


# In[210]:


region_stats.values.sum()


# In[211]:


len(region_stats[region_stats>100])


# In[212]:


len(region_stats)


# In[213]:


len(region_stats[region_stats<=100])


# In[214]:


region_stats_less_than_100 = region_stats[region_stats<=100]
region_stats_less_than_100


# In[215]:


len(X_train.regioncode.unique())


# In[216]:


X_train.regioncode = X_train.regioncode.apply(lambda x: 'other' if x in region_stats_less_than_100 else x) # #we are replacing the values with 'other' for those regions with less than 100 obs
len(X_train.regioncode.unique()) #now, we only have 43 unique regioncodes instead of 196


# In[217]:


X_train.head()


# In[218]:


X_train.shape


# We do the same encoding for the test dataset.
# 
# Note that we only do the last step because we want the test dataset to have the same dummies as the train dataset.

# In[219]:


X_test.neighborhoodcode = X_test.neighborhoodcode.apply(lambda x: 'other' if x in neighborhood_stats_less_than_100 else x)
len(X_test.neighborhoodcode.unique())


# In[220]:


X_test.regioncode = X_test.regioncode.apply(lambda x: 'other' if x in region_stats_less_than_100 else x)
len(X_test.regioncode.unique())


# For some reason, there is more than one neighborhoodcode and one regioncode in the test data compared to the train. I will just drop the extra dummy from the test dataset later while doing the data alignment.

# In[221]:


pd.set_option("display.max_rows", None, "display.max_columns", None) #I just checked all the obs of the test to make sure there's nothing weird going on.
print(X_test['neighborhoodcode'])


# Now we finally generate the dummies. Note that I will be dropping the first columns using drop_first=True to avoid the dummy variable trap.

# We generate the dummies for the train.

# In[222]:


X_train_d = X_train[potential]


# In[223]:


X_train_d = pd.get_dummies(X_train_d, columns=potential,drop_first=True)


# In[224]:


X_train_d.head()


# In[225]:


X_train_d.shape


# We generate the dummies for the test.

# In[226]:


X_test_d = X_test[potential]


# In[227]:


X_test_d = pd.get_dummies(X_test_d, columns=potential,drop_first=True)


# In[228]:


X_test_d.head()


# In[229]:


X_test_d.shape


# Our sub-DataFrame with dummies for the train dataset have 99 columns, while for the test, there are 105. This has occured because when we stored the neighborhoodcodes and regioncodes with less than 100 observations, some codes in neighborhoodcodes and regioncodes that should've been identified as other was for the test was not identified. This is just a small difference so we will just solve this by dropping the extra dummies while we align later.

# In[230]:


#we store the variable names of the dummy because this will be important later on when we are matching the number of columns in the train and test dataset
#we need to store the column names of these dummies because it will be gone once we convert it to numpy array or concatenate it later
X_train_d_columns = list(X_train_d.columns)
X_test_d_columns = list(X_test_d.columns)


# In[231]:


#just checking if the column names were stored properly
X_test_d_columns
X_train_d_columns


# In[232]:


print(len(X_train_d_columns))
print(len(X_test_d_columns))


# I store the year and tax year variables as we will drop them later when we generate polynomail features.

# In[233]:


X_train_year = X_train[['year', 'taxyear']]


# In[234]:


X_test_year = X_test[['year', 'taxyear']]


# *Creating Polynomial Features*

# I create a sub-DataFrame that will contain the features that I want to transform using polynomial features.
# 
# I do this for both the train and test dataset.

# In[235]:


X_train_f = X_train.drop(['heatingtype', 'citycode', 'countycode', 'neighborhoodcode', 'regioncode', 'year', 'taxyear'], axis = 1)


# In[236]:


X_train_f.head()


# In[237]:


#do the same for the test data
X_test_f = X_test.drop(['heatingtype', 'citycode', 'countycode', 'neighborhoodcode', 'regioncode', 'year', 'taxyear'], axis = 1)


# In[238]:


X_test_f.head()


# In[239]:


print(X_train_f.shape)
print(X_test_f.shape)

#we will be using the dataframes above later to generate features


# Now we finally generate the polynomial features.

# In[240]:


from sklearn.preprocessing import PolynomialFeatures as plf


# In[241]:


#first, let's try plf of order 2
#note that I'm not doing instantiate and fit in one go because when I tried it, there are having errors when I use the .get_feature_names method
order = 2
poly = plf(order)

phi_train = poly.fit_transform(X_train_f)
phi_test = poly.fit_transform(X_test_f)


# Note that I have tried having polynomial features of order 5, 4, and 3 respectively. But I couldn't fit the lasso without taking so much time, so I resort to 2. 
# 
# Further, for this dataset, I see no reason why there will be cubic and high order relationships for the features. But if I had more computing power, having higher-ordered features will be interesting if it wouldn't be zeroed out by an optimized Lasso model.

# In[242]:


#we compare the number of columns in before and after the plf (9 vs. 55)

print(X_train_f.shape)
print(X_test_f.shape)

print(phi_train.shape)
print(phi_test.shape)


# Notice that the dimensions of the features is 55 since polynomial features not only adds powers of each feature but also the interactions between them. Therefore, if we initial have 9 features, we do not just expect 9*2+1 in our new matrix phi_train.
# 
# The formula for calculating the number of the polynomial features is N(n,d)=C(n+d,d) where n is the number of the features, d is the degree of the polynomial, C is binomial coefficient(combination). So in this case, we have:

# In[243]:


import math

math.factorial(11)/(math.factorial(11-2)*math.factorial(2))


# In[244]:


def how_many_features(n,d):
    return math.factorial(n+d)/(math.factorial(n+d-d)*math.factorial(d))


# In[245]:


how_many_features(9,2)


# In[248]:


pd.DataFrame(phi_train).head()
#we observe that the features we have transformed trhu plf have no column names
#but we can get their names (as shown in the next cell)


# In[249]:


#let's store the names of these features because it's important for the test and train dataset alignment later
import numpy as np
phi_train_columns = np.array(poly.get_feature_names(X_train_f.columns))
phi_test_columns = np.array(poly.get_feature_names(X_test_f.columns))


# In[250]:


phi_train_columns


# In[251]:


#let's concatenate the columns names for both the dummies and the features, as well as the years columns
final_X_train_columns = np.concatenate([phi_train_columns, X_train_year.columns, X_train_d_columns])
final_X_test_columns = np.concatenate([phi_test_columns, X_test_year.columns, X_test_d_columns])


# In[252]:


print(final_X_train_columns.shape)
print(final_X_test_columns.shape)


# In[253]:


print(final_X_train_columns)
print(final_X_test_columns)


# We combine the generated dummies, years, and generated polynomial features to have our almost final datasets. (Almost final because we will have to do rescaling later). Note that this dataset contains a constant term.

# In[254]:


final_X_train = np.concatenate((phi_train, X_train_year, X_train_d), axis=1)
final_X_test = np.concatenate((phi_test, X_test_year, X_test_d), axis=1)


# In[255]:


final_X_train = pd.DataFrame(final_X_train)
final_X_test = pd.DataFrame(final_X_test)


# In[256]:


final_X_train.columns = final_X_train_columns
final_X_test.columns = final_X_test_columns


# In[257]:


pd.DataFrame(final_X_train).head()


# In[258]:


pd.DataFrame(final_X_test).head()


# In[259]:


print(final_X_train.shape)
print(final_X_test.shape)


# We align the train and test datasets.

# In[ ]:


final_X_train = pd.DataFrame(final_X_train)
final_X_test = pd.DataFrame(final_X_test)


# In[263]:


# Get missing columns in the training test
missing_cols = set( final_X_train.columns ) - set( final_X_test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    final_X_test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
final_X_test= final_X_test[final_X_train.columns]


# In[264]:


print(final_X_train.shape)
print(final_X_test.shape)


# ### Building the second model

# **Linear Regression using transformed features (no scaling yet)**

# In[320]:


regr  = LinearRegression(fit_intercept=False) ## fit_intercept = False as we already have it in phi earlier
regr.fit(final_X_train,y_train)


# In[321]:


y_hat_final = regr.predict(final_X_train)


# In[267]:


my_r2_plot(y_train, y_hat_final)


# Notice that 0.513 is better than 0.478 in-sample correlation before the variable transformations (polynomial features and dummies)

# In[322]:


##producing cross-validated predictions
y_hat_cv_final = cvp(regr, final_X_train, y_train, cv=80) # doing cv=80 because LOO takes too long

my_r2_plot(y_train, y_hat_cv_final)

# the value of R-squared is 0.360, there is some correlation between the y_train but not strong
#therefore go back to previous steps and optimize paramaters


# Also, note that the r-squared with cross-validated predictions is much lower than the usual in-sample r-squared (0.524 < 0.557). In other words, the performance of the model decreased drastically.
# 
# Notice also that the difference between the in-sample and cvp of our original dataset is smaller compared to the one with transform dataset  (0.485-0.489 = **0.06**) vs (0.513 - 0.383 =  **0.130**)
# 
# Therefore, if we transform are variables without penalizing (just the usual regression or alpha = 1, our model would be more inaccuarate compared to when there was no feature transformation.

# In[269]:


#Step 6:
print(regr.coef_)


# In[270]:


y_hat_test = regr.predict(final_X_test) #we produce predictions for our test dataset from our fitted model.


# In[ ]:


##Step 8:
#test_predictions_submit = pd.DataFrame({"lotid": test_original["lotid"], "parcelvalue": y_hat_test})
#test_predictions_submit.to_csv("test_predictions_submit.csv", index = False)


# In[ ]:


## we've seen that the score for this model perhaps could be very much improved so we use now lasso regression


# ### Standardization

# In[271]:


#drop the intercept before standardization
final_X_train = final_X_train.iloc[: , 1:]


# In[272]:


final_X_train.head()


# In[273]:


final_X_train.shape


# In[274]:


final_X_test = final_X_test.iloc[: , 1:]


# In[275]:


final_X_test.head()


# In[276]:


final_X_test.shape


# Next we use the standardization that Jack used in class. But this returns the error "Dataset may contain too large values. You may need to prescale your features." So I will use StandardScaler instead.

# In[ ]:


# standardisation of input is critical: We will use sklearn to do this

# generic lasso regression object
#from sklearn.preprocessing import scale as scl
#scaled_final_X_train = scl(final_X_train)


# In[277]:


##standardization of train data before lasso

from sklearn import preprocessing
# Get column names first
names = final_X_train.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_final_X_train = scaler.fit_transform(final_X_train)
scaled_final_X_train = pd.DataFrame(scaled_final_X_train, columns=names)


# In[278]:


scaled_final_X_train.head()


# In[279]:


scaled_final_X_train.shape


# In[280]:


##standardization of test data before lasso

# Get column names first
names = final_X_test.columns

# Fit your data on the scaler object
#We do not need to fit the objects again. 
#For sc, we want to keep the method we used to fit X_train_poly. 
#This means that the test data will not be perfectly standardised, and that is fine. 
#So instead of fit_transform, we use transform.
scaled_final_X_test = scaler.transform(final_X_test)
scaled_final_X_test = pd.DataFrame(scaled_final_X_test, columns=names)


# In[281]:


scaled_final_X_test.shape


# In[282]:


scaled_final_X_test.sample(10)


# I notice that the dummy columns are not with 0 and 1's anymore but they still contain binary values, so that's okay.

# In[283]:


print(type(scaled_final_X_train))
print(type(scaled_final_X_test))


# In[284]:


#I change the type/class to a numpy array because that would be faster
scaled_final_X_train_np = np.array(scaled_final_X_train)
scaled_final_X_test_np = np.array(scaled_final_X_test)
y_train_np = np.array(y_train)


# In[286]:


def print_shape(a):
    for i in a:
        print(i.shape)


# In[287]:


print_shape(scaled)


# ### Building the third model

# Note that all the proceeding models use Lasso, but with different ways how to find and set the hyperparameter alpha.

# **Lasso using transformed features**

# I'll use LassoLarsIC (whose results are based on AIC/BIC criteria) for faster computation.

# *Lasso using LassoLarsIC, using the Akaike Information Criterion (AIC)*

# In[288]:


from sklearn import linear_model


# In[289]:


regr_lasso = linear_model.LassoLarsIC(criterion="aic", normalize=False, max_iter = 100000)
regr_lasso.fit(scaled_final_X_train_np, y_train_np)
alpha_regr_lasso = regr_lasso.alpha_
print(alpha_regr_lasso)


# Now we do it with a graph to better visualize how the model was selected through the AIC. We also plot it agains BIC.

# In[290]:


import time

# This is to avoid division by zero while doing np.log10
EPSILON = 1e-4

# LassoLarsIC: least angle regression with BIC/AIC criterion

model_bic = linear_model.LassoLarsIC(criterion="bic", normalize=False)
t1 = time.time()
model_bic.fit(scaled_final_X_train_np, y_train_np)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = linear_model.LassoLarsIC(criterion="aic", normalize=False)
model_aic.fit(scaled_final_X_train_np, y_train_np)
alpha_aic_ = model_aic.alpha_


def plot_ic_criterion(model, name, color):
    criterion_ = model.criterion_
    plt.semilogx(
        model.alphas_ + EPSILON,
        criterion_,
        "--",
        color=color,
        linewidth=3,
        label="%s criterion" % name,
    )
    plt.axvline(
        model.alpha_ + EPSILON,
        color=color,
        linewidth=3,
        label="alpha: %s estimate" % name,
    )
    plt.xlabel(r"$\alpha$")
    plt.ylabel("criterion")


plt.figure()
plot_ic_criterion(model_aic, "AIC", "b")
plot_ic_criterion(model_bic, "BIC", "r")
plt.legend()
plt.title("Information-criterion for model selection (training time %.3fs)" % t_bic)


# In[291]:


print(alpha_bic_)
print(alpha_aic_)


# In[323]:


regr_lasso = linear_model.LassoLarsIC(criterion='aic', fit_intercept=True, max_iter=100000, normalize=False) #we've dropped the constant before scaling so we will fit the intercept
regr_lasso.fit(scaled_final_X_train,y_train)

#352 iterations, alpha=1.441e-06, previous alpha=1.333e-06, with an active set of 151 regressors.


# In[324]:


print(regr_lasso.alpha_)


# In[325]:


#Step 6
print(regr_lasso.coef_)


# We observe that some coefficients are zeroed out. This means that our model done feature selection and parameter tuning at the same time. Let's check how many features have been retained (and zeroed out using the code below.

# In[326]:


print("Total coefficiets:", len(regr_lasso.coef_))
print("Non-zero coefficiets:", np.count_nonzero(regr_lasso.coef_))


# There are now 136 non-zero coefficients. This means that 18 features have been zeroed out by our model.

# Which features have been zeroed out by lasso? Examples are 'numfireplace', 'roomnum', 'numbathnumbedroom', etc.

# In[327]:


dict_zip(final_X_train.columns, regr_lasso.coef_)


# In[328]:


y_hat_train = regr_lasso.predict(scaled_final_X_train) #we produce predictions from our fitted model based on train data


# In[329]:


#IN-SAMPLE
my_r2_plot(y_train, y_hat_train)

#we observe that this model has higer correlation / higher explanatory power than the OLS ( 0.573 vs. 0.485), 
#so this a good sign


# In[330]:


##producing cross-validated predictions
y_hat_cv = cvp(regr_lasso, scaled_final_X_train, y_train, cv=80) # doing cv=80 because LOO takes too long


# In[303]:


my_r2_plot(y_train, y_hat_cv)


#the cross-validated predictions worse, however (0.480 vs 0.421). The reduction in score is also higher.
#but still, our results aren't that bad!!
#however, when I submitted my predictions to Kaggle, it returned a really high NMSE


# In[304]:


y_hat_test = regr_lasso.predict(scaled_final_X_test_np) 
#the submitted predictions to kaggle returned a high negative mean square error: 841082.34786
#why? this is my worst performing model so far


# **Just some exploration**
# 
# Now checking again using just OLS, but now with the scaled data. This is because I'm wondering why my Lasso perfomed worse than the usual Linear Regression.

# In[305]:


from sklearn.linear_model import LinearRegression
regr = LinearRegression(fit_intercept=True)
regr.fit(scaled_final_X_train_np,y_train_np)
regr.score(scaled_final_X_train_np,y_train_np)

#the result is quite good


# In[306]:


##producing cross-validated predictions
y_hat_cv = cvp(regr, scaled_final_X_train, y_train, cv=40) # doing cv=40 because LOO takes too long

regr.score(scaled_final_X_train_np,y_hat_cv)

#Why is the usual regression doing really bad using scaled data under cross-validated predictions?


# ### Building the fourth model

# Lasso with some random alpha I've set. Our previous model used alpha=1.57. I'll use a smaller alpha this time.

# In[ ]:


regr_lasso = linear_model.Lasso(random_state = 0, max_iter=500000, alpha=0.001, tol=0.1, fit_intercept=True) 
#random state is setting seed for reproducible results


# In[ ]:


regr_lasso.fit(scaled_final_X_train_np, y_train_np)


# In[ ]:


#Step 6
print(regr_lasso.coef_)


# In[ ]:


print("Total coefficiets:", len(regr_lasso.coef_))
print("Non-zero coefficiets:", np.count_nonzero(regr_lasso.coef_))
#none of the coefficients have been zeroed out


# In[ ]:


## IN-SAMPLE ##
y_hat = regr_lasso.predict(scaled_final_X_train)
#print(y_hat.shape)

my_r2_plot(y_train, y_hat)


# In[ ]:


##producing cross-validated predictions
y_hat_cv = cvp(regr_lasso, scaled_final_X_train_np, y_train_np, cv=40) # doing cv=40 because LOO takes too long

print(r2_score(y_train,y_hat_cv)) #equals to 0.42229123280112824
print(regr_lasso.score(scaled_final_X_train_np,y_hat_cv)) # equals to #equals to 0.977778279489165


# In[ ]:


#now we try on test dataset
y_hat_test = regr_lasso.predict(scaled_final_X_test)

#kaggle returned an RMSE of 841082.34791


# ### Building the fifth model

# **GridSearchCV**

# Lasso again, with grid search of alphas this time. We include 1.57 and 0.001, the alphas we've used before.
# 
# I've tried many times, but oftentimes, the model does not converge, unless I set tol=1. However, upon checking the coefficients, I was not able to zero out anything. So I will change tol to a smaller value.However, I've run the search for 6 hours and it still has not finished.

# In[307]:


from sklearn.model_selection import cross_val_score
lasso = linear_model.Lasso(random_state = 0, max_iter=5000000, tol=1)


alphas = np.array([0.000007, 0.00008,0.00012,0.0005, 0.001,1.5736681638814658]) #we include the alpha we got from the AIC
alphas_long = np.array([0.000007, 0.00002, 0.00005,0.00008,0.00012,0.0002,0.0003,0.0005,0.0006,0.002])

tuned_parameters = [{'alpha': alphas}] ## dictionary


# In[308]:


# create a scorer to evaluate performance

from sklearn.metrics import mean_squared_error, make_scorer 

## ALWAYS read carefully documentation. copying here from make_scorer
## greater_is_better : boolean, default=True
# "Whether score_func is a score function (default), meaning high is 
# good, or a loss function, meaning low is good. 
# In the latter case, the scorer object will sign-flip 
# the outcome of the score_func.
mse = make_scorer(mean_squared_error,greater_is_better=False)


# In[309]:


from sklearn.model_selection import GridSearchCV

n_folds = 10 

clf = GridSearchCV(lasso, tuned_parameters, scoring = mse, 
                   cv=n_folds, refit=False)


# In[310]:


clf.fit(scaled_final_X_train, y_train)


scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
std_error = scores_std / np.sqrt(n_folds)


# In[311]:


# Extract best param
clf.best_params_

#the search identified the smallest alpha from the leastas the best one


# In[312]:


from sklearn.linear_model import Lasso
regr_lasso = Lasso(alpha=0.000007, random_state = 0, max_iter=3000000, tol=1)


# In[313]:


regr_lasso.fit(scaled_final_X_train_np,y_train_np)


# In[314]:


#Step 6
print(regr_lasso.coef_)


# In[315]:


## IN-SAMPLE ##
y_hat = regr_lasso.predict(scaled_final_X_train_np)
#print(y_hat.shape)

my_r2_plot(y_train, y_hat)


# In[316]:


y_hat_test = regr_lasso.predict(scaled_final_X_test_np)

#RMSE from Kaggle = 647076.20247


# **Some more exploration using Cross Validation**

# We compare cross validation scores from Ridge and Lasso

# In[317]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

for Model in [Ridge, Lasso]:
    model = Model()
    print('%s: %s' % (Model.__name__,
                      cross_val_score(model, scaled_final_X_train_np, y_train_np).mean()))


# In[331]:


alphas = np.logspace(-3, -1, 30)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), scaled_final_X_train_np, y_train_np, cv=3).mean()
            for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc='lower left')
plt.xlabel('alpha')
plt.ylabel('cross validation score')
plt.tight_layout()
plt.show()


# From the graph above, I conclude that the optimal alpha is something very small (close to 0). This is in line with the results of our GridSeachCV but not with LassoLarsIC using AIC.

# ### Building sixth model

# Let's try fitting the model using Ridge

# In[332]:


from sklearn.linear_model import Ridge

clf = Ridge(alpha=0.0000001)
clf.fit(scaled_final_X_train_np, y_train_np)


# In[333]:


y_hat_ridge = clf.predict(scaled_final_X_train_np)


# In[334]:


print(r2_score(y_train_np, y_hat_ridge))


# In[335]:


y_hat_ridge_cv = cvp(clf, scaled_final_X_train_np, y_train_np, cv=80)


# In[336]:


print(r2_score(y_train_np, y_hat_ridge_cv))


# In[337]:


y_hat_ridge_test = clf.predict(scaled_final_X_test_np)


# In[338]:


scaled_final_X_test.sample(10)


# In[339]:


print(scaled_final_X_train_np.shape)
print(scaled_final_X_test_np.shape)


# ***Other codes that I tried to do but took too long***

# In[ ]:


#GridSearcCV is taking too long, better to use RandomizedGridSearchInstead

#small
#cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=1)
#lasso_alphas = np.array([0.000007, 0.00005,0.00012, 0.00025,0.0005,0.002])
#lasso = Lasso(random_state=0,max_iter=100000)
#grid = dict()
#grid['alpha'] = lasso_alphas
#gscv = GridSearchCV( \
    #lasso, grid, scoring='neg_root_mean_squared_error', \
    #cv=50, n_jobs=-1)
#results = gscv.fit(scaled_final_X_train_df, y_train)

#print('MAE: %.5f' % results.best_score_)
#print('Config: %s' % results.best_params_)


# In[ ]:


#big
#cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
#lasso_alphas = np.array([0.000007, 0.00002, 0.00004, 0.00005,0.00008,0.0001,0.00012, 0.00015,0.0002,0.00025,0.0003,0.0004,0.0005,0.0006,0.0007,0.002])
#lasso = Lasso(random_state=0, max_iter = 1000000, tol=0.01)
#grid = dict()
#grid['alpha'] = lasso_alphas
#gscv = GridSearchCV( \
    #lasso, grid, scoring='neg_root_mean_squared_error', \
    #cv=50, n_jobs=-1)
#results = gscv.fit(scaled_final_X_train_new_df, y_train)

#print('MAE: %.5f' % results.best_score_)
#print('Config: %s' % results.best_params_)


# In[ ]:


#I've tried doing the lasso using the following parameters but it returned a worse fit that if only I did a simple linear regression
#further, there are some values that are negative
#let's try to force the coefficients to be positive


# In[ ]:


#lasso
#from sklearn.linear_model import Lasso
#alpha is what was lambda in our notation
#i'll change the max_iter recommended by the class notes to a smaller one since 1000000 it's taking too long 
#for my computer to fit the data
#regr_lasso = Lasso(alpha=0.0001, fit_intercept=False,warm_start=True,max_iter=100000)
#regr_lasso = Lasso(alpha=50, fit_intercept=False,warm_start=False,max_iter=50000, tol=0.1)
#regr_lasso = Lasso(alpha=5, fit_intercept=False,warm_start=False,max_iter=100000, tol=0.1)
#regr_lasso = Lasso(alpha=1, fit_intercept=False,warm_start=True,max_iter=10000)

regr_lasso = Lasso(alpha=1, fit_intercept=False, warm_start=True,max_iter=1000000, tol=0.1, positive=True)


# In[ ]:


#regr_lasso.fit(scaled_final_X_train_df,y_train)


# In[ ]:


#print(regr_lasso.coef_)


# In[ ]:


# Step 6: Report variable impact

# Report of the coefficients every after model fit


# In[ ]:


# Step 7: Prepare code to run and check performance of you model using a new input data with same exact format

#other step 7's are above (after the in-sample and cross validation predict)
y_hat_test = regr_lasso.predict(scaled_final_X_test_new_df) #we produce predictions from our fitted model based on test data


# ### Kaggle Predictions Submissions
# 
# Once you have produced testset predictions you can submit these to <i> kaggle </i> in order to see how your model performs. 
# 
# The following code provides an example of generating a <i> .csv </i> file to submit to kaggle
# 1) create a pandas dataframe with two columns, one with the test set "lotid"'s and the other with your predicted "parcelvalue" for that observation
# 
# 2) use the <i> .to_csv </i> pandas method to create a csv file. The <i> index = False </i> is important to ensure the <i> .csv </i> is in the format kaggle expects 

# In[ ]:


# Step 8: Produce .csv for kaggle testing 
test_predictions_submit = pd.DataFrame({"lotid": test_original["lotid"], "parcelvalue": y_hat_test})
test_predictions_submit.to_csv("test_predictions_submit.csv", index = False)

