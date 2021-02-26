import sys, os, matplotlib, time
import numpy as np
import pandas as pd
import statistics as st
from matplotlib import use
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import EllipticEnvelope as EE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest as IF
from sklearn.neighbors import LocalOutlierFactor as LOF
from umap import UMAP

# Read the file with the descriptors
dat=pd.read_csv('data.txt')
# Drop the column with the names of the structures
dat=dat.drop(columns=['Structure'])

# Perform a RobustScaler normalization
transformer = RobustScaler().fit(dat)
dat_transf=transformer.transform(dat)
print('\n-----------')
print('Data loaded')
print('-----------')
print('Number of samples:',dat.shape[0])
print('Number of features per sample:',dat.shape[1])


# Split the set into test and training
x_tr,x_ts=train_test_split(dat_transf, test_size=0.3, random_state=42)
if x_tr.shape[0] < dat.shape[1]**2:
    print('Remember the rule for outlier detection from covariance estimation:') 
    print('              n_samples > n_features ** 2')
print('\n--------------')
print('Data splitted') 
print('-------------')
print('Training set:',x_tr.shape[0])
print('Test set:', x_ts.shape[0])

'''
# UMAP dimensionality reduction from https://bit.ly/3aOdSZX (Freud readthedocs)
#ºººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººº
print('\n--------------------------------------')
print('Applying UMAP dimensionality reduction')
print('--------------------------------------')
t0=time.time()
umap=UMAP(random_state=42)
dat_reduced=umap.fit_transform(dat)
t1=time.time()

y=np.array([1,1,1,1,1,1,1,1,1,1,1])

plt.figure(figsize=(4, 3), dpi=300)
for i in range(max(y) + 1):
    indices = np.where(y == i)[0]
    plt.scatter(dat_reduced[indices, 0], dat_reduced[indices, 1],
                color=matplotlib.cm.tab10(i), s=8, alpha=0.2)
# It could be interesting to label the different structures here
#                label=list(structure_features.keys())[i])
plt.legend()
for lh in plt.legend().legendHandles:
    lh.set_alpha(1)
plt.savefig('umap.png')
print('Saved to umap.png')
print('%.2fs' % (t1-t0))
#ºººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººººº
'''

print('\n---------------')
print('Fitting the data')
print('----------------')
t0=time.time()
x_ee=EE(contamination=0.1,random_state=42).fit(x_tr)
t1=time.time()
print('Elliptic Envelope')
print('%.2fs\n' % (t1-t0))

t0=time.time()
x_if=IF(contamination=0.1,random_state=42).fit(x_tr)
t1=time.time()
print('Isolation Forest')
print('%.2fs\n' % (t1-t0))

t0=time.time()
x_lof=LOF(contamination=0.1).fit(x_tr)
t1=time.time()
print('Local Outlier Factor')
print('%.2fs\n' % (t1-t0))
