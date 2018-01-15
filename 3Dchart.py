import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#import dataframe
df = pd.read_csv('abalone.data')
df.columns = ['sex','length','diameter','height','whole weight','shucked weight','viscera weight','shell weight','rings']

#scale data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scale1 = df.drop(['sex'], axis=1)
scale2 = df.drop(['sex','rings'], axis=1)

scale1 = StandardScaler().fit_transform(scale1)
scale2 = StandardScaler().fit_transform(scale2)

#PCA transform for 3D scatter
pca = PCA(n_components=3, svd_solver='full')

A3 = pca.fit(scale1).transform(scale1)
B3 = pca.fit(scale2).transform(scale2)

#plt them
A3 = pd.DataFrame(A3)
B3 = pd.DataFrame(B3)

A3.columns = ['x','y','z']
B3.columns = ['x','y','z']

A3.drop(A3['z'].idxmax(), axis=0, inplace=True)
A3.drop(A3['z'].idxmax(), axis=0, inplace=True)
B3.drop(B3['y'].idxmax(),axis=0, inplace=True)
B3.drop(B3['y'].idxmax(),axis=0, inplace=True)

from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot')

fig = plt.figure()
plt.suptitle('with rings')
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(A3['x'], A3['y'], A3['z'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig = plt.figure()
plt.suptitle('no rings')
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(B3['x'], B3['y'], B3['z'])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()