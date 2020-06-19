import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm_notebook 
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
### DATA GENERATION###
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])
np.random.seed(0)
data, labels = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=0)
print(data.shape, labels.shape)
labels_orig = labels
labels = np.mod(labels_orig, 2)
plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap)
plt.show()
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0)
print(X_train.shape, X_val.shape)

##### NN CLASS #####

class FFnn:

  def __init__(self):
    self.w1  = np.random.randn()
    self.w2  = np.random.randn()
    self.w3  = np.random.randn()
    self.w4  = np.random.randn()
    self.w5  = np.random.randn()
    self.w6  = np.random.randn()
    self.w7  = np.random.randn()
    self.w8  = np.random.randn()
    self.w9  = np.random.randn()
    self.w10 = np.random.randn()
    self.w11 = np.random.randn()
    self.w12 = np.random.randn()
    self.w13 = np.random.randn()
    self.b1 = 0
    self.b2 = 0
    self.b3 = 0
    self.b4 = 0
    self.b5 = 0
    self.b6 = 0
    self.b7 = 0

  def sigmoid(self,x):

    return 1.0/(1.0 + np.exp(-x))

  def forward_pass(self,X):
    
    self.x1 , self.x2 = X
    self.a1 = self.w1*self.x1 + self.w3*self.x2 + self.b1
    self.h1 = self.sigmoid(self.a1)
    self.a2 = self.w2*self.x1 + self.w4*self.x2 + self.b2
    self.h2 = self.sigmoid(self.a2)

    self.a3 = self.w5*self.h1 + self.w8*self.h2 + self.b3
    self.h3 = self.sigmoid(self.a3)
    self.a4 = self.w6*self.h1 + self.w9*self.h2 + self.b4
    self.h4 = self.sigmoid(self.a4)
    self.a5 = self.w7*self.h1 + self.w10*self.h2 + self.b5
    self.h5 = self.sigmoid(self.a5)

    self.a6 = self.w11*self.h3 + self.w12*self.h4 + self.w13*self.h5 + self.b6
    self.h6 = self.sigmoid(self.a6)

    return self.h6

  def grad(self,x,y):
      self.forward_pass(x)

      self.dw11  = (self.h6 - y) * self.h6 *(1 - self.h6) * self.h3
      self.dw12  = (self.h6 - y) * self.h6 *(1 - self.h6) * self.h4
      self.dw13  = (self.h6 - y) * self.h6 *(1 - self.h6) * self.h2
      self.db6   = (self.h6 - y) * self.h6 *(1 - self.h6)
###################################################################################################
      self.dw5   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.h1
      self.dw6   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.h1
      self.dw7   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h5 * (1 - self.h5) * self.h1

      self.dw8   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.h2 
      self.dw9   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.h2
      self.dw10  =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h5 * (1 - self.h5) * self.h2

      self.db3   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h3 * (1 - self.h3)
      self.db4   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w12 * self.h4 * (1 - self.h4)
      self.db5   =  (self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h5 * (1 - self.h5)
####################################################################################################
      self.dw3   =  ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.w5 * self.h1 * (1 - self.h1) * self.x2) + ((self.h6 - y)  * self.h6 * (1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.w6 * self.h1 * (1 - self.h1) * self.x2)  + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w13 * self.h5 * (1 - self.h5) * self.w7 * self.h1 * (1 - self.h1) * self.x2)
      
      self.dw1   =  ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.w5 * self.h1 * (1 - self.h1) * self.x1) + ((self.h6 - y)  * self.h6 * (1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.w6 * self.h1 * (1 - self.h1) * self.x1)  + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w13 * self.h5 * (1 - self.h5) * self.w7 * self.h1 * (1 - self.h1) * self.x1)

      self.dw2   =  ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.w8  * self.h2 * (1 - self.h2) * self.x1) + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.w9  * self.h2 * (1 - self.h2) * self.x1) + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w13 * self.h5 * (1 - self.h5) * self.w10 * self.h2 * (1 - self.h2) * self.x1)

      self.dw4   =  ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.w8  * self.h2 * (1 - self.h2) * self.x2) + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.w9  * self.h2 * (1 - self.h2) * self.x2) + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w13 * self.h5 * (1 - self.h5) * self.w10 * self.h2 * (1 - self.h2) * self.x2)

      self.db1   =  ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.w5 * self.h1 * (1 - self.h1)) + ((self.h6 - y) * self.h6 * (1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.w6 * self.h1 * (1 - self.h1)) + ((self.h6 - y) * self.h6 *(1 - self.h6) * self.w13 * self.h5 * (1 - self.h5) * self.w7 * self.h1 * (1 - self.h1))

      self.db2   =  ((self.h6 - y) * self.h6 *(1 - self.h6) * self.w11 * self.h3 * (1 - self.h3) * self.w8  * self.h2 * (1 - self.h2)) + ((self.h6 - y) * self.h6 *(1 - self.h6) * self.w12 * self.h4 * (1 - self.h4) * self.w9  * self.h2 * (1 - self.h2)) + ((self.h6 - y) * self.h6 *(1 - self.h6) * self.w13 * self.h5 * (1 - self.h5) * self.w10 * self.h2 * (1 - self.h2))

   
  def fit(self,X,Y,epochs = 1,tr = 0.5,initialise = True, display_loss = False):

      if initialise:
        self.w1  = np.random.randn()
        self.w2  = np.random.randn()
        self.w3  = np.random.randn()
        self.w4  = np.random.randn()
        self.w5  = np.random.randn()
        self.w6  = np.random.randn()
        self.w7  = np.random.randn()
        self.w8  = np.random.randn()
        self.w9  = np.random.randn()
        self.w10 = np.random.randn()
        self.w11 = np.random.randn()
        self.w12 = np.random.randn()
        self.w13 = np.random.randn()
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0
        self.b4 = 0
        self.b5 = 0
        self.b6 = 0
        self.b7 = 0

      if display_loss:
        loss = {}
        lst = []

      for i in tqdm_notebook(range(epochs), total = epochs, unit ="epoch"):
        dw1,dw2,dw3,dw4,dw5,dw6,dw7,dw8,dw9,dw10,dw11,dw12,dw13,db1,db2,db3,db4,db5,db6 = [0]*19 
        for x,y in zip(X,Y):
            self.grad(x,y)  
            dw1  += self.dw1
            dw2  += self.dw2
            dw3  += self.dw3
            dw4  += self.dw4
            dw5  += self.dw5
            dw6  += self.dw6
            dw7  += self.dw7
            dw8  += self.dw8
            dw9  += self.dw9
            dw10 += self.dw10
            dw11 += self.dw11
            dw12 += self.dw12
            dw13 += self.dw13
            db6  += self.db6
            db5  += self.db5
            db4  += self.db4
            db3  += self.db3
            db2  += self.db2
            db1  += self.db1

        m = X.shape[1]
        self.w1  -= tr *dw1/m
        self.w2  -= tr *dw2/m
        self.w3  -= tr *dw3/m
        self.w4  -= tr *dw4/m
        self.w5  -= tr *dw5/m
        self.w6  -= tr *dw6/m
        self.w7  -= tr *dw7/m
        self.w8  -= tr *dw8/m
        self.w9  -= tr *dw9/m
        self.w10 -= tr *dw10/m
        self.w11 -= tr *dw11/m
        self.w12 -= tr *dw12/m
        self.w13 -= tr *dw13/m
        self.b6  -= tr *db6/m
        self.b5  -= tr *db5/m
        self.b4  -= tr *db4/m
        self.b3  -= tr *db3/m
        self.b2  -= tr *db2/m
        self.b1  -= tr *db1/m

        if display_loss:
          y_pred = self.predict(X)
          loss[i] = mean_squared_error(y_pred,Y)
          #print(loss[i])
      
      if display_loss:
          for q in loss.values():
              lst.append(q)  
          plt.plot(lst)
          plt.xlabel('Epochs')
          plt.ylabel('Mean Squared Error')
          plt.show()


  def predict(self,X):
     y_pred = []
     for x in X:
       y_pred.append(self.forward_pass(x)) 
     return y_pred      

####################################
ffn = FFnn()
ffn.fit(X_train, Y_train, epochs=2000, tr=.012, display_loss=True)  

Y_pred_train = np.array(ffn.predict(X_train))
Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()
Y_pred_val = np.array(ffn.predict(X_val))
Y_pred_binarised_val = (Y_pred_val >= 0.5).astype("int").ravel()
accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
accuracy_val = accuracy_score(Y_pred_binarised_val, Y_val)

print("Training accuracy", round(accuracy_train, 2))
print("Validation accuracy", round(accuracy_val, 2))

#NN CLASSIFICATION VISUALIZATION
plt.scatter(X_train[:,0], X_train[:,1], c=Y_pred_binarised_train, cmap=my_cmap, s=15*(np.abs(Y_pred_binarised_train-Y_train)+.2))
plt.show()