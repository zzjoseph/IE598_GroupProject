#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:07:58 2018

@author: siyangzhang
"""

from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/Users/siyangzhang/Desktop/课程/ie598 machine learning in finance/groupproject/MLF_GP1_CreditScore.csv")

df.isnull().any()

X, y = df.iloc[0:1700,0:26], df.InvGrd

print( X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)
print( X_train.shape, y_train.shape)

# Standardize the features

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print summary of data frame
summary = X.describe()
print(summary)



# Principal component analysis in scikit-learn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)



pca = PCA()
X_train_pca = pca.fit_transform(X_train)
pca.explained_variance_ratio_



#plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1])
#plt.xlabel('PC 1')
#plt.ylabel('PC 2')
#plt.show()

pca.explained_variance_ratio_.shape      

plt.bar(range(0, 26), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(0, 26), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()





pca = PCA(n_components=15)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train_pca,y_train)

print('Training accuracy:', lr.score(X_train_pca, y_train))
print('Test accuracy:', lr.score(X_test_pca, y_test))

#X_test=X_test_pca
#X_train=X_train_pca



#feat_labels = df[0:]
#
#importances = lr.feature_importances_
#indices = np.argsort(importances)[::-1]
#for f in range(X_train.shape[1]):
#    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
#plt.title('Feature Importance')
#plt.bar(range(X_train.shape[1]), importances[indices], align='center')
#plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
#plt.xlim([-1, X_train.shape[1]])
#plt.tight_layout()
#plt.show()



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA

# ## LDA via scikit-learn




lda = LDA(n_components=15)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)
print('Training accuracy:', lr.score(X_train_lda, y_train))
print('Test accuracy:', lr.score(X_test_lda, y_test))




#plot_decision_regions(X_train_lda, y_train, classifier=lr)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()
#
#
#plot_decision_regions(X_test_lda, y_test, classifier=lr)
#plt.xlabel('LD 1')
#plt.ylabel('LD 2')
#plt.legend(loc='lower left')
#plt.tight_layout()
#plt.show()



# ## Kernel principal component analysis in scikit-learn

kpca = KernelPCA(n_components=15, kernel='poly', gamma=10)
X_train_kpca = kpca.fit_transform(X_train, y_train)
X_test_kpca = kpca.transform(X_test)

lr = LogisticRegression()
lr = lr.fit(X_train_kpca, y_train)
print('Training accuracy:', lr.score(X_train_kpca, y_train))
print('Test accuracy:', lr.score(X_test_kpca, y_test))

from sklearn import datasets,decomposition,manifold


decomposition.IncrementalPCA

def plot_KPCA(*data):
    X,y = data
    kernels = ['linear','poly','rbf','sigmoid']
    fig = plt.figure()

    for i,kernel in enumerate(kernels):
        kpca = decomposition.KernelPCA(n_components=15, kernel=kernel)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2, 2, i+1)
        for label in np.unique(y):
            position = y == label
            ax.scatter(X_r[position,0],X_r[position,1],label="target=%d"%label)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.legend(loc='best')
            ax.set_title('kernel=%s'% kernel)
    plt.suptitle("KPCA")
    plt.show()
plot_KPCA(X, y)

#Fit a logistic classifier model and print accuracy score


lr = LogisticRegression(penalty='l1', C=1.0)
lr.fit(X_train, y_train)
print('Training accuracy:', lr.score(X_train, y_train))
print('Test accuracy:', lr.score(X_test, y_test))





