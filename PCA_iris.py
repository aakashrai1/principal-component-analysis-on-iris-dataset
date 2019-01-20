# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 03:49:27 2018
@author: akash
"""

# Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Reading the dataset and loading it in pandas dataframe
df = pd.read_csv('iris-data.csv', sep = ',', names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class'])

# Separating feature attributes and class attribute
featuresDf = df.loc[:, ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']]

# Normalizing feature attribute data
featuresDf = StandardScaler().fit_transform(featuresDf)

## PCA Code

# Calculating mean of column values (features)
meanVals = np.mean(featuresDf, axis=0)

# Calculating covariance of the feature matrix by subtracting the mean values
covarianceMatrix = (featuresDf - meanVals).T.dot((featuresDf - meanVals)) / (featuresDf.shape[0] - 1)

# Calculating Eigenvalues and Eigenvectors which satisfy the equation
eValues, eVec = np.linalg.eig(covarianceMatrix)

# Creating a pair of eigenvalues and eigenvectors and sorting them in descending order
eigenPairs = [(np.abs(eValues[i]), eVec[:, i]) for i in range(len(eValues))]
eigenPairs.sort()
eigenPairs.reverse()

# Extracting top 2 features that have max eigenvalues
top2Matrix = np.hstack((eigenPairs[0][1].reshape(4,1), eigenPairs[1][1].reshape(4,1)))

# Generating top 2 principal components
principalComp = featuresDf.dot(top2Matrix)

# Creating a new pandas dataframe and using new pca attribute values generated after transformation
pcaDF = pd.DataFrame(data = principalComp, columns = ['pca1', 'pca2'])
newDf = pd.concat([pcaDF, df[['class']]], axis = 1) # axis = 1 is for columns

# Plotting the graph
fig = plt.figure(figsize = (7,7))
axis = fig.add_subplot(111, facecolor='white')
axis.set_xlabel('PCA 1', fontsize = 12)
axis.set_ylabel('PCA 2', fontsize = 12)
axis.set_title('PCA on Iris dataset', fontsize = 15)

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
for classVal, color in zip(classes, ['y', 'b', 'r']):
    indexVals = newDf['class'] == classVal
    axis.scatter(newDf.loc[indexVals, 'pca1'], newDf.loc[indexVals, 'pca2'], c = color, s = 50)
axis.legend(classes, loc="upper right")
axis.grid(linewidth=0.5)
fig.savefig('PCA_fig.png', dpi=200)

## Looking at the plotted graph we can see that Iris-setosa is easily distinguishable from other two classes ##
