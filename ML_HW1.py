#!/usr/bin/env python
# coding: utf-8

# (a) loading data 

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
import numpy as np 



from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


dt = pd.read_csv('C:/Users/simeon/Desktop/ML_HW1_data.csv')
dt=pd.DataFrame(dt)
print(dt.head())


# (b) pre processing Exploratory and data analysis 

# i. 

# In[5]:



print(sns.lmplot( x="pelvic_incidence", y="pelvic_tilt", data=dt, fit_reg=False, hue='class1', legend=True))

print(sns.lmplot( x="lumbar_lordosis", y="sacral_slope", data=dt, fit_reg=False, hue='class1', legend=True))
plt.ylim(-50,150)
print(sns.lmplot( x="pelvic_radius", y="spondyloisthesis", data=dt, fit_reg=False, hue='class1', legend=True))
plt.ylim(-50,150)


# ii.

# In[7]:


fig, ax = plt.subplots(2,1)

sns.boxplot(x="class1", y="pelvic_incidence", data=dt, order=["Normal", "Abnormal"],ax=ax[0])


sns.boxplot(x="class1", y="pelvic_tilt", data=dt, order=["Normal", "Abnormal"],ax=ax[1])
plt.show

fig, ax = plt.subplots(2,1)

sns.boxplot(x="class1", y="lumbar_lordosis", data=dt, order=["Normal", "Abnormal"],ax=ax[0])


sns.boxplot(x="class1", y="sacral_slope", data=dt, order=["Normal", "Abnormal"],ax=ax[1])
plt.show

fig, ax = plt.subplots(2,1)

sns.boxplot(x="class1", y="pelvic_radius", data=dt, order=["Normal", "Abnormal"],ax=ax[0])


sns.boxplot(x="class1", y="spondyloisthesis", data=dt, order=["Normal", "Abnormal"],ax=ax[1])
plt.show


# iii. creating training and testing sets

# In[8]:


a = dt[dt.class1 =="Normal"]

a = np.array(a)
tr_no = a[0:70,]
a = dt[dt.class1 =="Normal"]

a = np.array(a)
tst_no = a[70:310,]


a = dt[dt.class1 =="Abnormal"]

a = np.array(a)
tr_ab = a[0:140,]

a = dt[dt.class1 =="Abnormal"]

a = np.array(a)
tst_ab = a[140:310,]
train = np.concatenate([tr_no, tr_ab])

test = np.concatenate([tst_no, tst_ab])

#train = pd.DataFrame(train)
#train.columns = ['pelvic_incidence',	'pelvic_tilt',	'lumbar_lordosis',	'sacral_slope',	'pelvic_radius',	'spondyloisthesis',	'class1']

x_train = train[:,0:6]
y_train = train[:,6]
x_test = test[:,0:6]
y_test =test[:,6]
#######################


# (c) Classification using KNN on Vertebral Column Data Set
# 
# i.& ii.

# In[12]:


train_error = list(range(208))
test_error = list(range(208))

for K in range(208):
 K_value = K+1
 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric = 'euclidean')
 neigh.fit(x_train, y_train) 
 y_pred_test = neigh.predict(x_test)
 print( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
 test_error[K] = 1- accuracy_score(y_test, y_pred_test)

for K in range(208):
 K_value = K+1
 neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric = 'euclidean')
 neigh.fit(x_train, y_train) 
 y_pred_train = neigh.predict(x_train)
 print( "error rate is ", 1- accuracy_score(y_train,y_pred_train),"% for K-Value:",K_value)
 train_error[K] = 1- accuracy_score(y_train, y_pred_train)


K_value = range(208)
#plotting results 
plt.ylim(0,.4)
plt.plot( K_value, train_error, marker='', markerfacecolor='blue', markersize=1, color='blue', linewidth=1)
plt.plot( K_value, test_error,  marker='', color='green', linewidth=1)
plt.title('accuracy with various K values')
plt.xlabel("K value")
plt.ylabel('accuracy in %')
print(plt.legend())



# we can see that our highest accuracy and lowest error occurs at a k value of 3, with accuracy = 81% test and 84% train however the highest total accuracy i which we add the two values together
#happens when k=1 with accuracy = 74% test and 100% train, to test we will try both of these values for K 






neigh = KNeighborsClassifier(n_neighbors = 3, weights='uniform', algorithm='auto', metric = 'euclidean')

neigh.fit(x_train, y_train) 
labels = ['Abnormal', 'Normal']
y_pred_train = neigh.predict(x_train)
y_pred_test = neigh.predict(x_test)
print(confusion_matrix(y_train, y_pred_train, labels))


print(confusion_matrix(y_test, y_pred_test))



#both output identical confusion matrixes 
#therefore we will say k = 3 is our k*
#true positive rate for train   aka recall
print(59/(59+11))
#true positive rate for test   aka recall
print(23/(23+7))

# true negative rate for train
print(11/(11+11))
# true negative rate for test
print(7/(7+1))
# precision for train
print(59/(59+11))
# precision for test
print(23/(23+1))

#F-score for train 
print(2*((0.8428571428571429*.8428571428571429)/(0.8428571428571429+.8428571428571429)))

#F-score for test
print(2*((.9583333333333334*.7666666666666667)/(.9583333333333334+.7666666666666667)))


# we can see that our highest accuracy and lowest error occurs at a k value of 3, with accuracy = 92% test and 84% train and corresponding errors of 1-accuracy 
# 

# iii.

# In[14]:


N = list(range(10, 211, 10))
plot_error = []
for i in N:
    tr_no1 = tr_no[0:round(i/3),:]
    tr_ab1 = tr_ab[0:(i-round(i/3)),:]
    train = np.concatenate([tr_no1, tr_ab1])
    x_train = train[:,0:6]
    y_train = train[:,6]
    num =int( i/10)
   
   
    test_error= []
    for K in list(range(1, i,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric = 'euclidean')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        ( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
        
    neigh = KNeighborsClassifier(n_neighbors = k_use, weights='uniform', algorithm='auto', metric = 'euclidean')
    neigh.fit(x_train, y_train) 
    y_pred_test = neigh.predict(x_test)
    bwb2 = ( 1- accuracy_score(y_test, y_pred_test))
    plot_error.append(bwb2)
    
    
plt.plot( N, plot_error, marker='', markerfacecolor='blue', markersize=1, color='blue', linewidth=1)
plt.title('accuracy with various trainging sample sizes')
plt.xlabel("training sample size")
plt.ylabel('error')  
    


# as we can see, the larger the training sample the better our model, with some exceptions occuring between a sample size of 30 and 100

# D. Replace the Euclidean metric with the following metrics
# and test them. Summarize the test errors 

# In[19]:


test_error = []
for K in list(range(1, 196,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric = 'minkowski')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        print( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = int((test_error.index(min(test_error))))
        k_use = (minpos*5)+1
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='uniform', algorithm='auto', metric = 'minkowski')
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)
 
test_error_minowski = 1- accuracy_score(y_test, y_pred_test)


# Manhattan
test_error = []
for K in list(range(1, 196,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric = 'manhattan')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        ( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='uniform', algorithm='auto', metric = 'manhattan')
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)
 
test_error_manhattan = 1- accuracy_score(y_test, y_pred_test)



#minoswki with various logbase(10) values 
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='uniform', algorithm='auto', metric = 'minkowski', p=1)
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)

test_error = [] 
test_error_minowski_log = []
for i in np.linspace(.1,1,10):
    print(i)
    neigh = KNeighborsClassifier(n_neighbors = k_use, weights='uniform', algorithm='auto', metric = 'minkowski', p = 10**i)
    neigh.fit(x_train, y_train) 
    y_pred_test = neigh.predict(x_test)
    test_error_minowski_log.append(1- accuracy_score(y_test, y_pred_test))

# the best log10 P is .1 or .3 
# Chebyshev
test_error = []
for K in list(range(1, 196,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto', metric = 'chebyshev')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        ( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='uniform', algorithm='auto', metric = 'chebyshev')
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)
 
test_error_cheb = 1- accuracy_score(y_test, y_pred_test)

#mahalanobis

from sklearn.datasets import make_classification
from sklearn.neighbors import DistanceMetric
x = x_train
y = y_train
xt = x_train
yt = y_train
test_error = []
for K in list(range(1, 99,5)):
        x, y = make_classification()
        xt, yt = make_classification()
        DistanceMetric.get_metric('mahalanobis', V=np.cov(x))
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors =K,algorithm='brute',  metric='mahalanobis', metric_params={'V': np.cov(x)})
        neigh.fit(x,y)
        y_pred_test = neigh.predict(xt)
        last = []
        for i in range(len(y)):
            if y[i] != y_pred_test[i]:
                last.append(i)
        accuracy = len(last)/len(y_pred_test)
        bwb = ( 1- accuracy)
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
  
#optimal k = 4      

x = x_train
y = y_train
xt = x_train
yt = y_train
test_error = []
for K in list(range(1, 99,5)):
        x, y = make_classification()
        xt, yt = make_classification()
        DistanceMetric.get_metric('mahalanobis', V=np.cov(x))
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors =4,algorithm='brute',  metric='mahalanobis', metric_params={'V': np.cov(x)})
        neigh.fit(x,y)
        y_pred_test = neigh.predict(xt)
        last = []
        for i in range(len(y)):
            if y[i] != y_pred_test[i]:
                last.append(i)
        accuracy = len(last)/len(y_pred_test)
        bwb = ( 1- accuracy)
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1

plotable_y = [.099,.10999,.10999,.10999,.47 ]
plotable_x =[1,2,3,4,5]
plotable_x_lab = {'1 minowski','2 manhattan','3 log10 = .1','4 chebyshev','5 mahalanobis'}

plt.plot( plotable_x, plotable_y, marker='', markerfacecolor='blue', markersize=1, color='blue', linewidth=1)
plt.title('accuracy with various metrics')
plt.xlabel("metric")
plt.ylabel('error') 
print('1 minowski','2 manhattan','3 log10 = .1','4 chebyshev','5 mahalanobis')


# the best errors that were achieved occured with our various types was with the minowki type in which we achieved an error of .0999

# e)
# The majority polling decision can be replaced by weighted decision, in which the
# weight of each point in voting is
# inversely proportional
# to its distance from the
# query/test data point. In this case, closer neighbors of a query point will have
# a greater influence than neighbors which are further away. Use weighted voting
# with Euclidean, Manhattan, and Chebyshev distances and report the best test
# errors 

# In[21]:


test_error = []
for K in list(range(1, 196,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='distance', algorithm='auto', metric = 'euclidean')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        ( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='distance', algorithm='auto', metric = 'euclidean')
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)
 
test_error_euclidean = 1- accuracy_score(y_test, y_pred_test)





test_error = []
for K in list(range(1, 196,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='distance', algorithm='auto', metric = 'manhattan')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        ( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='distance', algorithm='auto', metric = 'manhattan')
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)
 
test_error_manhattan = 1- accuracy_score(y_test, y_pred_test)

test_error = []
for K in list(range(1, 196,5)):
        K_value = K
        neigh = KNeighborsClassifier(n_neighbors = K_value, weights='distance', algorithm='auto', metric = 'chebyshev')
        neigh.fit(x_train, y_train) 
        y_pred_test = neigh.predict(x_test)
        ( "error rate is ", 1-accuracy_score(y_test,y_pred_test),"% for K-Value:",K_value)
        bwb = ( 1- accuracy_score(y_test, y_pred_test))
        test_error.append(bwb)
        minpos = test_error.index(min(test_error))
        k_use = (minpos*5)+1
        
neigh = KNeighborsClassifier(n_neighbors = k_use, weights='distance', algorithm='auto', metric = 'chebyshev')
neigh.fit(x_train, y_train) 
y_pred_test = neigh.predict(x_test)
 
test_error_cheb = 1- accuracy_score(y_test, y_pred_test)  
 
print(test_error_euclidean)
print(test_error_manhattan)
print(test_error_cheb)


# we can see that these weights did improve the accuracy of the manhattan type but the others remained the same 
# 

# (f)
# What is the lowest training error rate you achieved in this homework?

# the lowest test error rate that i got was with a euclidean type and a k value of 3 and uniform weights  0.07999999999999996.
# the lowest train error rate that i got was zero, this occured with the same parameters but with a k value of 1 

# In[ ]:




