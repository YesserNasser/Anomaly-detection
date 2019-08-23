# -*- coding: utf-8 -*-
"""
Created on Fri Dec  14 10:49:40 2018
@author: Yesser H. Nasser
"""
'''
here we explore an anomaly detection algorithm called an Isolation Forest.
the algorithm can be applied to univriate or mutivariate datasets.
isolated forest algorithm has one parameter (rate) which controls the target rate of anomaly detection.

since isolated forest algorithm can handle multivariate data, it is ideal for detecting anomalies when you have mutliple input features.
in our case, our input features will be the trading volume for a list of ETF symbols.

Symbols to consideer in this study are:
'SPY': SPDR S&P 500 ETF Trust, is used to track the S&P 500 stock market index 
'IWM': iShares Russell 2000 Index, is used by day traders and investors alike to gain access to the small-cap segment of US stocks. It is highly liquid.
'DIA': SPDR Dow Jones Industrial Average ETF, a price-weighted index of 30 large-cap US stocks, selected by the editors of the Wall Street Journal.
'IEF': iShares Barclays 7-10 Year Trasry Bnd Fd, tracks a market-value-weighted index of debt issued by the US Treasury with 7-10 years to maturity remaining.
'TLT': iShares 20+ Year Treasury Bond ETF, Tracks the Barclays U.S. 20+ Year Treasury Bond Index.
'GLD': Tracks the gold spot price, less expenses and liabilities, using gold bars held in London vaults.
'SLV': SLV tracks the silver spot price, less expenses and liabilities, using silver bullion held in London.
'USD': us dollar

this application aims at evaluating when the trading volume for our list of symbols as a whole is in 
an anomalous state. this could mean, for example that we are dtecting a spike in trading volume.

'''
import pandas as pd
from pandas_datareader import data as web 
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import time

# =============================== usinig yahoo finance API ===========================================================
symbols = ['SPY','IWM','DIA','IEF','TLT','GLD','SLV','USD',] 

start = dt.datetime(2012,1,1)
end = dt.datetime(2018,11,30)

volume= []
closes = []
for symbol in symbols:
    print (symbol)
    vdata = web.DataReader(symbol, 'yahoo', start, end)
    cdata= vdata[['Close']]
    closes.append(cdata)
    vdata = vdata[['Volume']]
    volume.append(vdata)
volume = pd.concat(volume, axis=1).dropna()
volume.columns = symbols
closes = pd.concat(closes, axis=1).dropna()
closes.columns = symbols

print(volume.head())
print(closes.head())

volume.plot(figsize=(12,6))
plt.ylabel('Volume')
plt.show()

'''
the time series of volume has significant spikes in trading volume accross ETF
universe. the anomaly detection algorith will try to identify these anomalies based on 
trading volume.
'''
# =========================================== Isolation Forest =======================================
# perfomr anoumaly detction on volume
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
V = preprocessing.scale(volume)
C = preprocessing.scale(closes)
Iso_forest1 = IsolationForest(max_samples='auto', contamination=0.08, n_estimators=10)
Iso_forest1.fit(V)
Iso_forest1.fit(C)

volume['Vanomaly'] = Iso_forest1.predict(V)
closes['Canomaly'] = Iso_forest1.predict(C)

volume_anomaly = volume.loc[volume['Vanomaly']==-1,['SPY','IWM','DIA','IEF','TLT','GLD','SLV','USD']]
closes_anomaly = closes.loc[closes['Canomaly']==-1,['SPY','IWM','DIA','IEF','TLT','GLD','SLV','USD']]

volume_init = volume.drop('Vanomaly',1)
closes_init = closes.drop('Canomaly' ,1)


plt.figure(figsize=(12,6))
volume.plot(figsize=(12,6))
plt.plot(volume_init.SPY)
plt.scatter(volume_anomaly.index,volume_anomaly['SPY'],c='red')

# ================================================ Further visualization ============================================
# histogram of volumes
volume_vis = volume
plt.figure(figsize=(12,6))
for x in volume_vis.columns:
    plt.hist(volume_vis[x])
plt.ylabel('Frequency')
plt.xlabel('Volume')

# find the optimum number of clusters of data  
from sklearn.cluster import KMeans

n_cluster = range(1,20)
kmeans = [KMeans(n_clusters=i).fit(volume_vis) for i in n_cluster]
scores = [kmeans[i].score(volume_vis) for i in range(len(kmeans))]

plt.figure(figsize=(10,6))
plt.plot(n_cluster, scores, 'o')
plt.xlabel('Number of clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.grid(1)
plt.show()

from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(volume_vis.values)

kmeans = KMeans(n_clusters = 10)
y = kmeans.fit_predict(X)
volume_vis['cluster'] = y

from sklearn.decomposition import PCA
# pricipal component analysis
pca=PCA(n_components=2)
principalComponents=pca.fit_transform(X)
ReducedData=pd.DataFrame(principalComponents, columns=['pca1','pca2'])
ReducedData['cluster']=y

# ploting the data
fig=plt.figure(figsize=(8,8))
plt.scatter(ReducedData.pca1, ReducedData.pca2, c=kmeans.labels_.astype(float), s=50, Alpha=0.5)
plt.xlabel('Principal Component 1', fontsize=15)
plt.ylabel('Principal Component 2', fontsize=15)
plt.title('Principal Components Analysis (2)', fontsize=20)
plt.grid()
