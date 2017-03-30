# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:52:30 2016

@author: chill
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 17:59:10 2016

@author: chill
"""

import numpy as np
import scipy as sp
import csv
import math
from sklearn import svm
from sklearn.svm import SVC
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
input = open('C:\\Users\\chill\\Desktop\\fall2016\\Artificial intelligence-2\\assignments\\project\\bank-additional-full.txt')
i=0
#foormatting input data to rows and columns
inputmatrix= [[0 for x in range(21)] for y in range(41190)]
for row in csv.reader(input):
    #splitting each row
 inputmatrix[i]=row[0].split(";")
 i=i+1
 
inputmatrix=np.array(inputmatrix)
# a=np.cov(inputmatrix)

tlabels=inputmatrix[1:,20]
#classifing true and false and choosing them equally
labels=inputmatrix[1:,20]

datalabels=inputmatrix[0,:20]
labels=np.array(labels)
inputmatrix=inputmatrix[1:,:20]

undeletedcol=np.array([0,1,2,3,5,6,7,8,9,11,13,14,15,16,17,18])
datalabels=datalabels[undeletedcol]

#input2matrix=inputmatrix[]
inputmatrix=inputmatrix[:,undeletedcol]

    
    
    
    
#converting nonnumeric to numeric
t=np.where(inputmatrix[:,1]=='"admin."')
inputmatrix[t,1]=11
t=np.where(inputmatrix[:,1]=='"blue-collar"')
inputmatrix[t,1]=1
t=np.where(inputmatrix[:,1]=='"entrepreneur"')
inputmatrix[t,1]=2
t=np.where(inputmatrix[:,1]=='"housemaid"')
inputmatrix[t,1]=3
t=np.where(inputmatrix[:,1]=='"management"')
inputmatrix[t,1]=4
t=np.where(inputmatrix[:,1]=='"retired"')
inputmatrix[t,1]=5
t=np.where(inputmatrix[:,1]=='"self-employed"')
inputmatrix[t,1]=6
t=np.where(inputmatrix[:,1]=='"services"')
inputmatrix[t,1]=7
t=np.where(inputmatrix[:,1]=='"student"')
inputmatrix[t,1]=8
t=np.where(inputmatrix[:,1]=='"technician"')
inputmatrix[t,1]=9
t=np.where(inputmatrix[:,1]=='"unemployed"')
inputmatrix[t,1]=10
t=np.where(inputmatrix[:,1]=='"unknown"')
inputmatrix[t,1]=0

t=np.where(inputmatrix[:,2]=='"divorced"')
inputmatrix[t,2]=3
t=np.where(inputmatrix[:,2]=='"married"')
inputmatrix[t,2]=1
t=np.where(inputmatrix[:,2]=='"single"')
inputmatrix[t,2]=2
t=np.where(inputmatrix[:,2]=='"unknown"')
inputmatrix[t,2]=0

t=np.where(inputmatrix[:,3]=='"basic.4y"')
inputmatrix[t,3]=7
t=np.where(inputmatrix[:,3]=='"basic.6y"')
inputmatrix[t,3]=1
t=np.where(inputmatrix[:,3]=='"basic.9y"')
inputmatrix[t,3]=2
t=np.where(inputmatrix[:,3]=='"high.school"')
inputmatrix[t,3]=3
t=np.where(inputmatrix[:,3]=='"illiterate"')
inputmatrix[t,3]=4
t=np.where(inputmatrix[:,3]=='"professional.course"')
inputmatrix[t,3]=5
t=np.where(inputmatrix[:,3]=='"university.degree"')
inputmatrix[t,3]=6
t=np.where(inputmatrix[:,3]=='"unknown"')
inputmatrix[t,3]=0

#t=np.where(inputmatrix[:,4]=='"no"')
#inputmatrix[t,4]=2
#t=np.where(inputmatrix[:,4]=='"yes"')
#inputmatrix[t,4]=1
##t=np.where(inputmatrix[:,4]=='"unknown"')
##inputmatrix[t,4]=0

t=np.where(inputmatrix[:,4]=='"no"')
inputmatrix[t,4]=2
t=np.where(inputmatrix[:,4]=='"yes"')
inputmatrix[t,4]=1
t=np.where(inputmatrix[:,4]=='"unknown"')
inputmatrix[t,4]=0

t=np.where(inputmatrix[:,5]=='"no"')
inputmatrix[t,5]=2
t=np.where(inputmatrix[:,5]=='"yes"')
inputmatrix[t,5]=1
t=np.where(inputmatrix[:,5]=='"unknown"')
inputmatrix[t,5]=0


t=np.where(inputmatrix[:,6]=='"cellular"')
inputmatrix[t,6]=0
t=np.where(inputmatrix[:,6]=='"telephone"')
inputmatrix[t,6]=1
t=np.where(inputmatrix[:,6]=='"unknown"')
inputmatrix[t,6]=0



t=np.where(inputmatrix[:,7]=='"jan"')
inputmatrix[t,7]=0
t=np.where(inputmatrix[:,7]=='"feb"')
inputmatrix[t,7]=1
t=np.where(inputmatrix[:,7]=='"mar"')
inputmatrix[t,7]=2
t=np.where(inputmatrix[:,7]=='"apr"')
inputmatrix[t,7]=3
t=np.where(inputmatrix[:,7]=='"may"')
inputmatrix[t,7]=4
t=np.where(inputmatrix[:,7]=='"jun"')
inputmatrix[t,7]=5
t=np.where(inputmatrix[:,7]=='"jul"')
inputmatrix[t,7]=6
t=np.where(inputmatrix[:,7]=='"aug"')
inputmatrix[t,7]=7
t=np.where(inputmatrix[:,7]=='"sep"')
inputmatrix[t,7]=8
t=np.where(inputmatrix[:,7]=='"oct"')
inputmatrix[t,7]=9
t=np.where(inputmatrix[:,7]=='"nov"')
inputmatrix[t,7]=10
t=np.where(inputmatrix[:,7]=='"dec"')
inputmatrix[t,7]=11

t=np.where(inputmatrix[:,8]=='"mon"')
inputmatrix[t,8]=0
t=np.where(inputmatrix[:,8]=='"tue"')
inputmatrix[t,8]=1
t=np.where(inputmatrix[:,8]=='"wed"')
inputmatrix[t,8]=2
t=np.where(inputmatrix[:,8]=='"thu"')
inputmatrix[t,8]=3
t=np.where(inputmatrix[:,8]=='"fri"')
inputmatrix[t,8]=4



t=np.where(inputmatrix[:,11]=='"failure"')
inputmatrix[t,11]=0
t=np.where(inputmatrix[:,11]=='"nonexistent"')
inputmatrix[t,11]=1
t=np.where(inputmatrix[:,11]=='"success"')
inputmatrix[t,11]=2



mask=np.random.rand(len(labels))<0.9
mask2=np.logical_not(mask)

#choosingtrain data
t1labels=labels[mask]
traindata=inputmatrix[mask]
traintarget=tlabels[mask]

traindata=np.array(traindata)
#choosingtestdata
t2labels=labels[mask2]
testdata=inputmatrix[mask2]
testtarget=tlabels[mask2]
testdata=np.array(testdata)

#converting target values to numeric data
traintarget=np.array(traintarget)
dl=np.where(traintarget=='"yes"')
traintarget[dl]=1
d2=np.where(traintarget=='"no"')
traintarget[d2]=0


testtarget=np.array(testtarget)
dl=np.where(testtarget=='"yes"')
testtarget[dl]=1
d2=np.where(testtarget=='"no"')
testtarget[d2]=0



traintarget=np.array(traintarget).astype(float)
traindata=np.array(traindata).astype(float)
testdata=np.array(testdata).astype(float)

covariances=np.zeros((16))
for i in range(16):
    s=traindata[:,i]
    s=np.array(s)
    covariances[i]=np.cov(s,traintarget)[0][1]
negcoc=np.where(covariances<0)
covariances[negcoc]=-covariances[negcoc]
B=sorted(range(len(covariances)),key=lambda x:covariances[x],reverse=True)
C=sorted(range(len(covariances)),key=lambda x:B[x])
a=np.array(([2,4,6,7,9,10,11,16]))
accuracy_nb=np.zeros((8))
accuracy_svm=np.zeros((8))
import random
#Cvalue=random.sample(range(1e-5, 100), 8)
Cvalue=np.array([0.008,0.08,0.01,0.6,0.2,0.1,1,1.01])
for i in range(8):
    columns=B[0:16]    
    #data using correlation
    corelatedtraindata=traindata[:,columns]
    traintarget=np.array(traintarget).astype(float)
    corelatedtraindata=np.array(corelatedtraindata).astype(float)
    #testdata
    
    corelatedtestdata=testdata[:,columns]   
    corelatedtestdata=np.array(corelatedtestdata).astype(float)
    
    testtarget=np.array(testtarget).astype(float)
    
    t=traintarget
    X = corelatedtraindata
    y = traintarget
    clf = SVC()
    clf.fit(X, y)  
    SVC(C=Cvalue[i], cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=False, random_state=None, shrinking=True,
        tol=0.001, verbose=False)
    testpredict=clf.predict(corelatedtestdata)
    accuracy_svm[i]=accuracy_score(testtarget,testpredict)    
    
    


print(accuracy_nb,'accur')
#testdata=testdata[:,0:6]
##reducing to two dimensions

pca=decomposition.PCA(n_components=2)
pca.fit(testdata)
tr=pca.transform(testdata)

la=np.where(traintarget==0)
ba=np.where(traintarget==1)
#x=tr[la]
#y=tr[ba]


print(accuracy_nb)


#plotting accuracy for svm

plt.title('svmaccuracy for various covariance')
plt.plot(a,accuracy_svm)
plt.xlabel('total number of predicted variables')
plt.ylabel('accuracy')
plt.show() 




correctlabels=np.where(testpredict==testtarget)
wronglabels=np.where(testpredict!=testtarget)
testcorrectlabels=testtarget[correctlabels]
testcorrectdata=tr[correctlabels]
testwrongdata=tr[wronglabels]
testwronglabels=testtarget[wronglabels]
correct1labels=np.where(testcorrectlabels==1)
correct0labels=np.where(testcorrectlabels==0)
wrong1labels=np.where(testwronglabels==1)
wrong0labels=np.where(testwronglabels==0)
testcorrect1labels=testcorrectdata[correct1labels]
testcorrect0labels=testcorrectdata[correct0labels]
testwrong1labels=testwrongdata[wrong1labels]
testwrong0labels=testwrongdata[wrong0labels]
#plt.plot(tr)
plt.title('NaivebayesClassification with 2 predictor variables')
plt.scatter(testcorrect1labels[:,0],testcorrect1labels[:,1],color='r',label='correctlyclassified yes')
plt.scatter(testcorrect0labels[:,0],testcorrect0labels[:,1],color='b',label='correctly classified no')
plt.scatter(testwrong1labels[:,0],testwrong1labels[:,1],color='g',label='incorrectly classified yes')
plt.scatter(testwrong0labels[:,0],testwrong0labels[:,1],color='y',label='incorrectly classified no' )
plt.xlim([-40,40])
#plt.plot(a,accuracy_nb)
#plt.xlabel('total number of predictor variables')
#plt.ylabel('accuracy')
#plt.plot()
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(Cvalue,accuracy_svm)