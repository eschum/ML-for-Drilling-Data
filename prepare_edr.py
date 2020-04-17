#File: prepare_EDR.py
#Name: Eric Schumacker
#Description: class to prepare EDR Data. 
#
#Use this class to abstract away all the data processing tasks. Moving into the modeling phase, 
#we want to take the training data as a given; and hide all the data processing through
#good encapsulation.

#This will eventually be placed in prepare_EDR.py; currently developing in ipynb for ease of
#data visualization.

import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler

class PrepareEDR:
  def __init__(self, path=None):
    if path != None:
    
      #input raw data. 
      with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        self.headers = next(reader)
        data_input = np.array(list(reader))
    data = data_input.astype(float)
    
    #Select features (from experimentation and final feature deceisions in preprocessing.ipynb)
  
    #Add a feature (Bit depth) / (Hole Depth) that serves as a ratio or a 'locator' of where we are 
    #during trip.
    depth_ratio = data[:,1]/data[:,0]
    dr_col = np.expand_dims(depth_ratio, axis=1)
    self.X = np.hstack((dr_col, data))
    self.headers.insert(0, "Bit Depth / Hole Depth")

    #Remove all negative values from Feature 7 (differential Pressure)
    diff = self.X[:,7]
    diff[diff<0] = 0
    self.X[:,7] = diff

    #Delete feature 10 (On Bottom Hours), as we don't want to keep that feature at all.
    self.X = np.delete(self.X, [10], 1)
    self.headers = [x for x in self.headers if x not in ["On Bottom Hours"]]

    #Note - from here, we want to keep the original data; and separately build the training data
    #That way we can provide he classifications to the original data and make sense of it.

    #Delete features 1 and 2 from the training data. 
    self.X_dr = np.delete(self.X, [1,2], 1)
    self.headers_dr = [x for x in self.headers if x not in ["Hole Depth", "Bit Depth"]]

    #Transform Block height feature into Block Movement
    block_height = np.zeros(self.X.shape[0])
    block_height[0] = 0
    dh = np.zeros(self.X.shape[0])
    dh[0] = 0
    for i in range(1, self.X.shape[0]):
      dh[i] = self.X[i,6] - self.X[i-1,6]
      if dh[i] < 1 and dh[i] > -1:
        block_height[i] = 0
      elif dh[i] > 1:
        block_height[i] = 1
      else:
        block_height[i] = -1

    self.X_dr[:,4] = block_height
    self.headers_dr[4] = "Block Movement: + / - / 0"

    #Remove outliers - get indices of outliers to remove.
    highWeightOutliers = np.where(self.X_dr[:,2] > 50)
    diffSpikeOutliers = np.where(self.X_dr[:,5] > 2400)



    #Don't combine the removal operations, but instead distinctly remove both sets of outliers. 
    #Do it this way for code readability and reuse. 
    self.X = np.delete(self.X, highWeightOutliers, 0)
    self.X = np.delete(self.X, diffSpikeOutliers, 0)
    self.X_dr = np.delete(self.X_dr, highWeightOutliers, 0)
    self.X_dr = np.delete(self.X_dr, diffSpikeOutliers, 0)
    


  def getOriginalData(self):
    return (self.X_dr, self.headers_dr)

  def getOriginalDF(self):
    return pd.DataFrame(data = self.X, columns = self.headers)

  def getClusteringTrainingData(self):
    X_dr_scaler = StandardScaler()
    X_train_dr = X_dr_scaler.fit_transform(self.X_dr)
    return (X_train_dr, self.headers_dr, X_dr_scaler)

  #Implement a method that provides labeled training data for ROP optimization



