#Importing libraries

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm_notebook 
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
from numpy.linalg import norm

#Gen instantiation
np.random.seed(0)
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","yellow","green"])

#Load dataset

iris=load_iris()
data = iris.data[:,:]  
labels = iris.target

plt.scatter(data[:,0], data[:,1], c=labels, cmap=my_cmap)
plt.show()

print("Data shape",data.shape)
print("Labels shape",labels.shape)

X_train, X_val, Y_train, Y_val = train_test_split(data, labels, stratify=labels, random_state=0,test_size=0.2)
print(X_train.shape, X_val.shape, labels.shape)

enc = OneHotEncoder()
y_OH_train = enc.fit_transform(np.expand_dims(Y_train,1)).toarray()
y_OH_val = enc.transform(np.expand_dims(Y_val,1)).toarray()
print(y_OH_train.shape, y_OH_val.shape)

#Neural network class

class Rknet:
  def __init__(self,hidden_neurons = 2, initialization = "Xavier",activation_fn = "Leaky_relu", leaky_slope = 0.2 ):

    self.params             = {}
    self.all_layers         = 2  # hidden + o/p
    self.layer_sizes        = [4,hidden_neurons,3]
    self.activation_fn      = activation_fn
    self.leaky_slope        = leaky_slope
    self.gradients          = {}
    self.update_params      = {}
    self.prev_update_params = {}
    np.random.seed(0)

    if initialization  == "Random":
      for i in range(1,self.all_layers+1):
        self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])
        self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])

    elif initialization == "Xavier":
      for i in range(1,self.all_layers+1):
        self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(1/self.layer_sizes[i-1])
        self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])

    elif initialization == "He":
      for i in range(1,self.all_layers+1):
        self.params["W"+str(i)] = np.random.randn(self.layer_sizes[i-1],self.layer_sizes[i])*np.sqrt(2/self.layer_sizes[i-1])
        self.params["B"+str(i)] = np.random.randn(1,self.layer_sizes[i])

    for i in range(1,self.all_layers):
      self.update_params["V_W"+str(i)]      = 0
      self.update_params["V_B"+str(i)]      = 0

      self.update_params["M_W"+str(i)]      = 0              # For Momentum based gradient descent and others
      self.update_params["M_B"+str(i)]      = 0

      self.prev_update_params["V_W"+str(i)] = 0         
      self.prev_update_params["V_B"+str(i)] = 0

  def forward_activation(self,X):
     if self.activation_fn == "sigmoid":
       return 1/(1.0 + np.exp(-X))

     if self.activation_fn == "tanh":
       return np.tanh(X)

     if self.activation_fn == "Relu":
       return np.maximum(0,X)

     if self.activation_fn == "Leaky_relu":
       return np.maximum(self.leaky_slope*X,X)

  def gradient_activation(self,X):
          
     if self.activation_fn == "sigmoid":
       return X*(1-X)

     if self.activation_fn == "tanh":
       return (1-np.square(X))

     if self.activation_fn == "Relu":
       return 1.0*(X>0)

     if self.activation_fn == "Leaky_relu":
       derv = np.zeros_like(X)
       derv[X<=0] = 0
       derv[X>0]  = self.leaky_slope
       return derv

  def accuracy(self):
     
     y_train_pred   = self.predict(X_train)
     y_val_pred     = self.predict(X_val)
     y_train_pred   = np.argmax(y_train_pred,1)
     y_val_pred     = np.argmax(y_val_pred,1)
     accu_train     = accuracy_score(y_train_pred,Y_train)
     accu_val       = accuracy_score(y_val_pred,Y_val)

     return accu_train,accu_val

  def softmax(self,X):

     return np.exp(X)/np.sum(np.exp(X),axis = 1).reshape(-1,1)

  def forward_pass(self,X):

     self.A1 = np.matmul(X,self.params["W1"]) + self.params["B1"]       # Nx2 * 2xlayer_sizes[Hidden] == Nxlayer_sizes[Hidden]
     self.H1 = self.forward_activation(self.A1)                         # Nxlayer_sizes[Hidden]
     #print("shape:{}".format(self.H1.shape))
     self.A2 = np.matmul(self.H1,self.params["W2"]) + self.params["B2"] # Nxlayer_sizes[Hidden] * layer_sizes[Hidden]*3 == Nx3
     self.H2 = self.softmax(self.A2)                                    # Nx3
     #print("shape:{}".format(self.H2.shape))

     return self.H2

  def grad(self,X,Y,params = None):
     
     if params is None:
      params = self.params

     self.forward_pass(X)
     m = X.shape[0]
     self.gradients["dA2"] = self.H2 - Y                                                      #Nx4
     self.gradients["dW2"] = np.matmul(self.H1.T, self.gradients["dA2"])                      #2xN * Nx4  == 2x4
     self.gradients["dB2"] = np.sum(self.gradients["dA2"],axis = 0).reshape(1,-1)             #Nx4        == 1x4
     self.gradients["dH1"] = np.matmul(self.gradients["dA2"],self.params["W2"].T)             #Nx4 * 4xlayer_size[Hidden] == Nxlayer_size[Hidden]
     self.gradients["dA1"] = np.multiply(self.gradients["dH1"],self.gradient_activation(self.H1)) # Nxlayer_size[Hidden] .* Nxlayer_size[Hidden] == Nxlayer_size[Hidden]
     self.gradients["dW1"] = np.matmul(X.T,self.gradients["dA1"])                             # 2xN .* Nxlayer_size[Hidden] == 2xlayer_size[Hidden]
     self.gradients["dB1"] = np.sum(self.gradients["dA1"],axis = 0).reshape(1,-1)                #Nxlayer_size[Hidden] == 1xlayer_size[Hidden]


  def fit(self,X,Y,epochs=100,learning_rate=0.1,algo = "GD",l2_norm = False,lambda_val = 0.8,display_loss = False,eta = 1):
     train_accuracies = {}
     val_accuracies   = {}
     
     if display_loss:
       trainz = []
       valz = []
       loss = []
       weight_mag = []
      
     for epoch in tqdm_notebook(range(epochs),total=epochs,unit ="epoch"):
       m = X.shape[0]
       self.grad(X,Y)
       for i in range(1,self.all_layers+1):
         if l2_norm:
           self.params["W"+str(i)] -= (eta*lambda_val)/m * self.params["W"+str(i)]   + eta * (self.gradients["dW"+str(i)]/m)
         else:
           self.params["W"+str(i)] -= eta * (self.gradients["dW"+str(i)]/m)  
         self.params["B"+str(i)]   -= eta * (self.gradients["dB"+str(i)]/m)  

       train_accuracy,val_accuracy=self.accuracy()
       train_accuracies[epoch]=train_accuracy
       val_accuracies[epoch]=val_accuracy
       if display_loss:
         Y_pred = self.predict(X)
         loss.append(log_loss(np.argmax(Y, axis=1), Y_pred))
         weight_mag.append((norm(self.params["W1"]) + norm(self.params["W2"]) + norm(self.params["B1"]) + norm(self.params["B2"]))/18)     

     if display_loss:    
          for k in train_accuracies.values():
            trainz.append(k)
          for z in val_accuracies.values():
            valz.append(z)   

          plt.plot(trainz,label="Train accuracy")
          plt.plot(valz,label="Validation accuracy")
          plt.plot(np.ones((epochs, 1))*0.9)
          plt.plot(np.ones((epochs, 1))*0.33)
          plt.xlabel('Epochs')
          plt.ylabel('Accuracy')
          plt.legend()
          plt.show()
      
          fig, ax1 = plt.subplots()
          color = 'tab:red'
          ax1.set_xlabel('epochs')
          ax1.set_ylabel('Log Loss', color=color)
          ax1.plot(loss, '-o', color=color)
          ax1.tick_params(axis='y', labelcolor=color)
          ax2 = ax1.twinx()  
          color = 'tab:blue'
          ax2.set_ylabel('Weight Magnitude', color=color)  # we already handled the x-label with ax1
          ax2.plot(weight_mag, '-*', color=color)
          ax2.tick_params(axis='y', labelcolor=color)
          fig.tight_layout()  
          plt.show()

  
  def predict(self, X):
    Y_pred = self.forward_pass(X)
    return np.array(Y_pred).squeeze()                
  
def print_accuracy():    
    Y_pred_train = model.predict(X_train)
    Y_pred_train = np.argmax(Y_pred_train,1)
    Y_pred_val = model.predict(X_val)
    Y_pred_val = np.argmax(Y_pred_val,1)
    accuracy_train = accuracy_score(Y_pred_train, Y_train)
    accuracy_val = accuracy_score(Y_pred_val, Y_val)
    print("Training accuracy", round(accuracy_train, 4))
    print("Validation accuracy", round(accuracy_val, 4))


model = Rknet(hidden_neurons=64)
model.fit(X_train, y_OH_train, epochs=550, eta=0.1, l2_norm=True, lambda_val=10, display_loss=True)
print_accuracy()    