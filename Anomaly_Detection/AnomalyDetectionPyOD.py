# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:40:46 2018

@author: Yesser H. Nasser
"""
'''
Angle-based outlier detection (ABOD): it considers the relationship between each point 
and its neighbors. ABOD performs well on multi-dimensional data. 
There are two different versions of ABOD: FAST ABOD: uses k-nearest neighbors 
to approximate, Original ABOD: considers all training points with high-time complexity.

k- Nearest Neighbors Detector: For any data point, the distance to its kth nearest 
neighbor is considered as the outlying score: PyOD support three kNN detectors:
        - Largest: uses the distance of the kth neighbor as the outlier score.
        - Mean: uses the average of all k neighbors as the outlier score
        - Median: uses the median of the distance to k neighbors as the outlier score.
        
Isolation Forest (IForest): it uses sklearn library. In this method data 
partitioning is done using a set of trees. isolation forest provides an anomaly 
score looking at how isolated the point is in the structure. 
The anomaly score is then used to identify outliers from normal observations 
isolation forest performs well on multi-dimensional data

Histogram-based outliers detection (HBOS): it is an effective unsupervised 
method which assumes the feature independence and calculates the outlier 
score by building histogram. It is much faster than multivariate approaches, but with less precision

Local Correlation Integral(LOCI): is very effective for detecting outliers 
and groups of outliers. It provides a LOCI plot for each point which summarizes
a lot of the information about the data in the area around the point, 
determining clusters, micro-clusters, their diameters and their inter-cluster distances.
  
Feature Bagging: Fits several base detectors on various sub-samples of the dataset.

Clustering Based Local Outlier Factor: It classifies the data into small clusters 
and large clusters. the anomaly score is then calculated based on the size of 
the cluster the point belongs to, as well as the distance to the nearest large cluster.  
  
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
# use ABOD and KNN
from pyod.models.abod import ABOD
# Histogram-base Outlier Detection
from pyod.models.hbos import HBOS

from pyod.models.feature_bagging import FeatureBagging

from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF

from pyod.models.knn import KNN

from pyod.models.iforest import IForest
# create a random data set
from pyod.utils.data import generate_data, get_outliers_inliers

# generate random data set with 2 features
X_train, Y_train = generate_data(n_train=300, train_only=True, n_features = 2)
# by default outlier fraction is 0.1
random_state = np.random.RandomState(42)
outliers_fraction = 0.1

# store outliers and inliers in different numpy arrays
x_outliers, x_inliers = get_outliers_inliers(X_train, Y_train)

n_inliers = len(x_inliers)
n_outliers = len(x_outliers)

# separate the two features and use it to plot the data
F1 = X_train[:,[0]].reshape(-1,1)
F2 = X_train[:,[1]].reshape(-1,1) 

# create a meshgrid:
xx, yy = np.meshgrid(np.linspace(-10,10,200), np.linspace(-10,10,200))

# scatter plot
plt.scatter(F1,F2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(1)

# create a dictionary and add all the models that will be used to detect 
# anomalies
classifiers = {
        'Angle-based Outlier Detector (ABOD)': ABOD(contamination=outliers_fraction),
        'Cluster-based Local Outlier Factor (CBLOF)':CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        'Feature Bagging':FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state),
        'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction,random_state=random_state),
        'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
        'Average KNN': KNN(method='mean',contamination=outliers_fraction)
        }


# set the sigure size:

plt.figure(figsize=(20,40))

for i, (clf_name,clf) in enumerate (classifiers.items()):
    
    # fit the dataset to the model
    clf.fit(X_train)
    
    # predict raw anomaly score
    score_pred = clf.decision_function(X_train)*-1
    
    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X_train)
    
    # no of erros in prediction 
    n_errors = (y_pred != Y_train).sum()
    print('No of Errors: ', clf_name, n_errors)
    
    # visualization 
    
    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(score_pred, 100*outliers_fraction)
    
    #decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])*-1
    Z = Z.reshape(xx.shape)
    
    subplot = plt.subplot(4,2,i+1)
    
    # fill blue colormap from minimu anomaly score to threshold value
    subplot.contour(xx,yy,Z, levels=np.linspace(Z.min(),threshold, 10), cmap = plt.cm.Blues_r)
    
    #darw red contour line where anomaly score is equal to the threshold
    a = subplot.contour(xx,yy, Z, levels=[threshold], linewidths=2, colors='red')
    
    #fill orange contor lines where range of anomaly score is from threshold to maximum anomaly score
    subplot.contourf(xx,yy,Z, levels=[threshold, Z.max()], colors='orange')
    
    
    # scatter plot of inliers with white dots
    b = subplot.scatter(X_train[:-n_outliers,0], X_train[:-n_outliers,1], c='white', s=20, edgecolor='k')
    
    #scatter plotof outliers with black dots
    c = subplot.scatter(X_train[-n_outliers:,0], X_train[-n_outliers:,1], c='black', s=20, edgecolor='k')
    subplot.axis('tight')
    
    subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop = matplotlib.font_manager.FontProperties(size=10),
            loc = 'lower right')
    subplot.set_title(clf_name)
    subplot.set_xlim((-10,10))
    subplot.set_ylim((-10,10))
plt.show()
plt.tight_layout()    
    
    
    











































































