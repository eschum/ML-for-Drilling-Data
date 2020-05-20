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
from sklearn.preprocessing import StandardScaler  #For clustering
from sklearn.preprocessing import MinMaxScaler    #For NLP
from sklearn.model_selection import train_test_split

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

    time_estimate = range(1, len(self.X) + 1)
    self.X = np.append(self.X, np.expand_dims(time_estimate, axis=1), 1)
    self.headers.append("Time Sequence")

    


  def getOriginalData(self):
    return (self.X_dr, self.headers_dr)

  def getDrDataFrame(self):
    return pd.DataFrame(data = self.X_dr, columns = self.headers_dr)

  def getOriginalDF(self):
    print("Executed")
    return pd.DataFrame(data = self.X, columns = self.headers)

  def getClusteringTrainingData(self):
    X_dr_scaler = StandardScaler()
    X_train_dr = X_dr_scaler.fit_transform(self.X_dr)
    return (X_train_dr, self.headers_dr, X_dr_scaler)

  #Method: getLateralData:
  #This function returns drilling data to/from a target depth. 
  #In practice, this would be replaced by the real-time acquition of data. 
  #We will train the model on a given interval of data (for instance, the first 1,000 feet of the
  #lateral) as a recommendation system for target drilling parameters to maintain.
  #
  #Output is tuple of (X_train, y_train, X_test, y_test, scaler)
  #The scaler is fit to only the training data. 
  #It is returned so the user can perform further operations on the data.
  def getLateralData(self, start, end):

    #we are only interested in the depth domain specified
    #Dimensionality reduction was further considered by removing all points where
    #bit was off bottom; and subsequently removing dimension of Bit Depth Ratio
    depthRangeToDelete = np.where((self.X[:,2] < start) | (self.X[:,2] > end) | (self.X[:,0] < 1.0))
    lateralData = np.delete(self.X_dr, depthRangeToDelete, axis=0)

    #Move ROP to the end so that we can do inverse transforms easier.
    ROP_data = lateralData[:,7]
    lateralData_labelsLast = np.delete(lateralData, [7], axis=1)
    lateralData_labelsLast = np.concatenate((lateralData_labelsLast, np.expand_dims(ROP_data, axis=1)), axis=1)
    
    #Next, split into training and testing. 
    train_set, test_set = train_test_split(lateralData_labelsLast, test_size=0.2, random_state = 42)

    #Normalize the data prior to splitting up.
    #Later, return this Scaler so that end users can perform inverse transform.
    #Move ROP to the end
    lateral_data_scaler = MinMaxScaler()  #Default is a range of [0,1]
    train_set_std = lateral_data_scaler.fit_transform(train_set)
    test_set_std = lateral_data_scaler.transform(test_set)

    
    #Remove the ROP data - index #7 - as the training labels.
    y_train = train_set_std[:,9]
    y_test = test_set_std[:,9]
    X_train = np.delete(train_set_std, [9], 1)
    X_test = np.delete(test_set_std, [9], 1)
    
    
    #No further dimensionality reduction was performed in order to maintain uniformity
    #With initial clustering data. 
    #return X_train, y_train, X_test, y_test, X_headers, y_header, scaler
    headers_X = [x for x in self.headers_dr if x not in ["On Bottom ROP"]]
    return (X_train, y_train, X_test, y_test, headers_X, ['ROP'], lateral_data_scaler)

  #helper function to create a lookback for time series prediction.
  def create_dataset(self, X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


  #Also get data over a depth interval, but prepare it for use with 
  #Keras time series layers. Also, to test, we are going to take the first
  #80% of the data as training data, and the last 20% of the data as testing data. 
  def getLateralTimeSeriesData(self, start, end, time_steps):
    train_split = 0.8
    #we are only interested in the depth domain specified
    depthRangeToDelete = np.where((self.X[:,2] < start) | (self.X[:,2] > end))
    lateralData = np.delete(self.X_dr, depthRangeToDelete, axis=0)

    #Move ROP to the end so that we can do inverse transforms easier.
    ROP_data = lateralData[:,7]
    lateralData_labelsLast = np.delete(lateralData, [7], axis=1)
    lateralData_labelsLast = np.concatenate((lateralData_labelsLast, np.expand_dims(ROP_data, axis=1)), axis=1)
    
    #Next, split into training and testing. 
    train_set = lateralData_labelsLast[0:int(len(lateralData_labelsLast)*train_split),]
    test_set = lateralData_labelsLast[int(len(lateralData_labelsLast)*train_split):,]

    #Normalize the data prior to splitting up.
    #Later, return this Scaler so that end users can perform inverse transform.
    #Move ROP to the end
    lateral_data_scaler = MinMaxScaler()  #Default is a range of [0,1]
    train_set_std = lateral_data_scaler.fit_transform(train_set)
    test_set_std = lateral_data_scaler.transform(test_set)

    
    #Remove the ROP data - index #7 - as the training labels.
    y_train = train_set_std[:,9]
    y_test = test_set_std[:,9]
    X_train = np.delete(train_set_std, [9], 1)
    X_test = np.delete(test_set_std, [9], 1)

    #Reshape the X and y arrays to have their "lookbacks"
    #This will reshape the matrix to [samples, time_steps back, n_features]
    #This is the required format for input into LSTM.
    time_steps = time_steps
    X_train_reshape, y_train_reshape = self.create_dataset(X_train, y_train, time_steps)
    X_test_reshape, y_test_reshape = self.create_dataset(X_test, y_test, time_steps)

    #No further dimensionality reduction was performed in order to maintain uniformity
    #With initial clustering data.
    #Note that we are also passing the original test data, so the client does not have to 
    #concatenate matrices and inverse transform, but instead can directly compare results.
    #A cleaner approach would be to write a separate method that can do this inverse_transform
    #and results comparison / appraisal.
    #return X_train, y_train, X_test, y_test, X_test_orig, y_test_orig, 
    #X_headers, y_header, scaler
    headers_X = [x for x in self.headers_dr if x not in ["On Bottom ROP"]]
    return (X_train_reshape, y_train_reshape, X_test_reshape, y_test_reshape, 
            X_test, headers_X, ['ROP'], lateral_data_scaler)

  
  #Method: Prepare Lateral Data From Clustering
  #Description: Here, we let the client cluster the data. 
  #They provide the clustered data (with column index #10 having the class names), and
  #this method then deletes all the other data, and returns the data to the client such that they
  #can directly use the data with another model (MLP, in our target use case)
  #Note also that this differs from prepareLateralData because after using the classes to filter data, 
  #we will then go ahead and delete the class
  def prepareLateralDataFromClustering(self, data, className):
    #we are only interested in the depth domain specified
    #Dimensionality reduction was further considered by removing all points where
    #bit was off bottom; and subsequently removing dimension of Bit Depth Ratio
    lateralData = data.to_numpy()
    depthRangeToDelete = np.where(lateralData[:,10] != className) 
    lateralData = np.delete(lateralData, depthRangeToDelete, axis=0)

    #Move ROP to the end so that we can do inverse transforms easier.
    #Delete the class column, as it has served its purpose, but now we want to remove this dimension.
    ROP_data = lateralData[:,7]
    lateralData_labelsLast = np.delete(lateralData, [7, 10], axis=1)
    lateralData_labelsLast = np.concatenate((lateralData_labelsLast, np.expand_dims(ROP_data, axis=1)), axis=1)
    
    #Next, split into training and testing. 
    train_set, test_set = train_test_split(lateralData_labelsLast, test_size=0.2, random_state = 42)

    #Normalize the data prior to splitting up.
    #Later, return this Scaler so that end users can perform inverse transform.
    #Move ROP to the end
    lateral_data_scaler = MinMaxScaler()  #Default is a range of [0,1]
    train_set_std = lateral_data_scaler.fit_transform(train_set)
    test_set_std = lateral_data_scaler.transform(test_set)

    
    #Remove the ROP data - index #7 - as the training labels.
    y_train = train_set_std[:,9]
    y_test = test_set_std[:,9]
    X_train = np.delete(train_set_std, [9], 1)
    X_test = np.delete(test_set_std, [9], 1)
    
    
    #No further dimensionality reduction was performed in order to maintain uniformity
    #With initial clustering data. 
    #return X_train, y_train, X_test, y_test, X_headers, y_header, scaler
    headers_X = [x for x in self.headers_dr if x not in ["On Bottom ROP", "Class"]]
    return (X_train, y_train, X_test, y_test, headers_X, ['ROP'], lateral_data_scaler)