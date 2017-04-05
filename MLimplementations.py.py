
from __future__ import division
import numpy as np
import scipy as sp
import csv
import math
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
testtarget=np.array(testtarget).astype(float)



ar=np.array([16])
accuracy_kmeans=0
accuracy_nb=[0 for  i in range(6)]
accuracy_svm=[0 for  i in range(6)]
#
#traindata11=traindata
#testdata11=testdata
#for im in range(1):



#finding covariancematrix
covariances=np.zeros((16))
#finding the top 6 columns with high covariances
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
for i in range(8):
    columns=B[0:a[i]]    
    #data using correlation
    corelatedtraindata=traindata[:,columns]
    traintarget=np.array(traintarget).astype(float)
    corelatedtraindata=np.array(corelatedtraindata).astype(float)
    #testdata
    
    corelatedtestdata=testdata[:,columns]   
    corelatedtestdata=np.array(corelatedtestdata).astype(float)
    
    testtarget=np.array(testtarget).astype(float)
    
    t=traintarget
    

validationaccuracy=[0 for i in range(4)]
validatesize=math.floor(len(traindata)/5)
#calculating k nearest neighbour
ksize=np.array([1,5,10,15])



#reducing traindata,train target to 2 components
pca=decomposition.PCA(n_components=2)
pca.fit(traindata)
traindata=pca.transform(testdata)

pca=decomposition.PCA(n_components=2)
pca.fit(testdata)
testdata=pca.transform(testdata)



##validatingdata to find the k nearest neighbour
for i in range(4):
    j=0
    kaccuracy=0    
    for j in range(4):
        if j==0 :
                validatetest=traindata[0:validatesize,:]
                validatetarget=traintarget[0:validatesize]
                
                Val_traindata=traindata[validatesize:,:]
               val_traintarget=traintarget[validatesize:]
       else:
               validatetest=traindata[j*validatesize:,:]

               validatetarget=traintarget[j*validatesize:]        
              val_traindata=traindata[:j*validatesize,:]
             val_traintarget=traintarget[:j*validatesize]
                if j==4:
               else:
                validatetest=traindata[validatesize*j:validatesize*(j+1),:]
               validatetarget=traintarget[validatesize*j:validatesize*(j+1)]
                
                traindata1=traindata[0:validatesize*j,:]
                traindata2=traindata[validatesize*(j+1):,:]
               traindata1=np.array(traindata1)
                traindata2=np.array(traindata2)

               val_traindata=np.vstack((traindata1,traindata2))                
            traintarget1=traintarget[0:validatesize*j]
                traintarget2=traintarget[validatesize*(j+1):]
                traintarget1=np.array(traintarget1)
                traintarget2=np.array(traintarget2)
              #val_traintarget=np.vstack((traintarget1,traintarget2))
##sp.sparse
#                
#        
#        
#        neigh = KNeighborsClassifier(n_neighbors=ksize[i])
#        neigh.fit(val_traindata,val_traintarget)
#        validate_predict=neigh.predict(validatetest)
#        accuracy=accuracy_score(validatetarget,validate_predict)
#        kaccuracy=accuracy+kaccuracy
#    validationaccuracy[i]=kaccuracy/5
#KNEAREST NEIGHBOURS WITH CORELATION GIVES MORE ACCURACY THAN OTHER

#bestK=np.argmax(validationaccuracy)
#neigh = KNeighborsClassifier(n_neighbors=ksize[bestK])

accuracy0=np.zeros((4))
accuracy1=np.zeros((4))
al=np.array([1,15,40,80])
indices=[1,15,40,100]
for i in range(4):
        traindata1=traindata
        testdata1=testdata
        neigh = KNeighborsClassifier(n_neighbors=al[i])
        neigh.fit(traindata1,traintarget)
        testpredict=neigh.predict(testdata1)
        accuracy_kmeans=accuracy_score(testtarget,testpredict)
        
        print(accuracy_kmeans,'accur')
        #testdata=testdata[:,0:6]
        ##reducing to two dimensions
        t1=testdata
        pca=decomposition.PCA(n_components=2)
        pca.fit(t1)
        tr=pca.transform(t1)
        
        totalzeros=np.where(testtarget==0)
        totalones=np.where(testtarget==1)
#x=tr[la]
#y=tr[ba]
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
        accuracy0[i]=len(correct0labels[0])/len(totalzeros[0])
        accuracy1[i]=2*len(correct1labels[0])/len(totalones[0])


#plt.plot(tr)
plt.title('knnclassification')
plt.scatter(testcorrect1labels[:,0],testcorrect1labels[:,1],color='r',label='correctlyclassified yes')
plt.scatter(testcorrect0labels[:,0],testcorrect0labels[:,1],color='b',label='correctly classified no')
plt.scatter(testwrong1labels[:,0],testwrong1labels[:,1],color='g',label='incorrectly classified yes')
plt.scatter(testwrong0labels[:,0],testwrong0labels[:,1],color='y',label='incorrectly classified no' )
#plt.xlim([-30,50])
plt.legend()
plt.tight_layout()
plt.show()


#Plotting the accuracy using various K nearest neighbours
#these probabilities are calculated by substituting selected values of k
y=np.array(([84.23,84.71,88.6,89.74,88.97,88.11]))
x=np.array([1,2,5,10,15,25])
plt.title('KNNaccuracies')
plt.plot(x,y)
plt.xlabel('nearest neighbours')
plt.ylabel('accuracies')
plt.show()



#indices = [5.5,6,7,8.5,8.9]
#Calculate optimal width
#barplot to show 0 and 1 accuracies
width =np.min(np.diff(indices))/3

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(indices-width,accuracy1,width,color='b',label='accuracyof1')
ax.bar(indices,accuracy0,width,color='r',label='accuracyof 0')
ax.set_xlabel(' nearestneighours(blue-accuracyofOnes ,red-accuracyofZeros)')
ax.set_ylabel('accuracy')
ax.set_title('histogram of accuracy for different N')
fig.tight_layout()
plt.show()
   

##plotting the accuracy various machinelearning algorithms
plt.title('accuracy for Ml algorithms')
plt.xlabel('correlated variables')
plt.ylabel('accuracy')
plt.plot(x,k,'b-o',label='kmeansalgorithm')
plt.plot(x,nb,'r-^',label='naivebayes')
##plt.plo
##plt.legend()
#
#
