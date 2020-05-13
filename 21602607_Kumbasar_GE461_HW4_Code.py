#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import plotly.graph_objects as go
import csv


# In[2]:


class NN:
    def __init__(self, train_x, train_y, test_x, test_y, hid_no, epoch, lr):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.hid_no = hid_no
        self.epoch = epoch
        self.lr = lr
        self.weight1 = np.random.uniform(low=0.0, high=10.0, size=(1, self.hid_no))
        self.weight2 = np.random.uniform(low=0.0, high=10.0, size=(self.hid_no,1))
        self.w1_hold = []
        self.w2_hold = []
        self.outputs = []
        self.train_rmse = []
        self.test_rmse = []
        self.output = 0
        
    def train(self):
        np.random.seed(5)
        
        for i in range(self.epoch):
            
            #print("Epoch Number: " + str(i+1))
            
            self.feedforward(self.train_x)
            summer = 0
            for j in range(self.train_y.shape[0]):
                summer += (self.output[j] - self.train_y[j])**2
            #summer = summer**(1/2)
            #print("MSE of train is: " + str(summer))
            self.train_rmse.append(summer)
            self.backprop()
            
            
            
            test_out = self.feedforward_test(self.test_x)
            
            summer_test = 0
            for k in range(self.test_y.shape[0]):
                summer_test = summer_test + (test_out[k] - self.test_y[k])**2
            self.test_rmse.append(summer_test)
                
            #print("MSE of test is: " + str(summer_test) + "\n")
        ret_train = self.do_feedforward(self.train_x)
        ret_test = self.do_feedforward(self.test_x)
        return np.array(self.test_rmse), np.array(self.train_rmse), ret_train, ret_test
            
    def get_rmse(self):
        
        return train_rmse, test_rmse
        
    def sigmoid(self, inp):
        sig = 1/(1 + np.exp(-inp))
        return sig
    
    def sigmoid_der(self, x):
        sig = self.sigmoid(x)
        der = sig * (1 - sig)
        return der
    
    def feedforward(self, inp):
        self.layer1 = self.sigmoid(np.dot(inp,self.weight1))
        self.output = self.sigmoid(np.dot(self.layer1, self.weight2))
        self.outputs.append(self.output)
        
    def do_feedforward(self, inp):
        layer1 = self.sigmoid(np.dot(inp,self.weight1))
        output = self.sigmoid(np.dot(layer1, self.weight2))
        return output
    
    def feedforward_test(self, inp):
        layer1 = self.sigmoid(np.dot(inp,self.weight1))
        output_test = self.sigmoid(np.dot(layer1, self.weight2))
        return output_test
        
        
    def backprop(self):
        
        self.w1_hold.append(self.weight1)
        self.w2_hold.append(self.weight2)
        out_der = self.sigmoid_der(self.output)


        
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_weights2 = np.dot(self.layer1.T, (2*(self.train_y - self.output) * out_der))
        
        delta1 = np.dot((2*(self.train_y - self.output) * out_der), self.weight2.T)
        
        delta_der = self.sigmoid_der(self.layer1)
        
        delta = delta1 * delta_der
        
        d_weights1 = np.dot(self.train_x.T, delta)

        self.weight1 += self.lr * d_weights1
        self.weight2 += self.lr * d_weights2
   


# In[3]:


class LinearReg:
    def __init__(self, train_x, train_y, test_x, test_y, epoch, lr):
        
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.epoch = epoch
        self.lr = lr
        self.weight1 = np.random.rand(1, 1)
        self.w1_hold = []
        self.outputs = []
        self.train_rmse = []
        self.test_rmse = []
        self.output = 0
        
    def train(self):

        
        for i in range(self.epoch):
            
            print("Epoch Number: " + str(i+1))
            
            self.feedforward(self.train_x)
            summer = 0
            for j in range(self.train_y.shape[0]):
                summer += (self.output[j] - self.train_y[j])**2
            #summer = summer**(1/2)
            print("MSE of train is: " + str(summer))
            self.train_rmse.append(summer)
            self.backprop()
            
            
            
            test_out = self.feedforward_test(self.test_x)
            summer_test = 0
            
            for k in range(self.test_y.shape[0]):
                summer_test = summer_test + (test_out[k] - self.test_y[k])**2
            self.test_rmse.append(summer_test)
                
            print("MSE of test is: " + str(summer_test) + "\n")
        
        
        return np.array(self.test_rmse), np.array(self.train_rmse)
            
    def get_rmse(self):
        
        return self.train_rmse, self.test_rmse
        
    def sigmoid(self, inp):
        sig = 1/(1 + np.exp(-inp))
        return sig
    
    def sigmoid_der(self, x):
        sig = self.sigmoid(x)
        der = sig * (1 - sig)
        return der
    
    def feedforward(self, inp):
        self.layer1 = self.sigmoid(np.dot(inp,self.weight1))
        self.output = self.layer1
        self.outputs.append(self.output)
        
    def feedforward_test(self, inp):
        layer1 = self.sigmoid(np.dot(inp,self.weight1))
        output_test = layer1
        return output_test
        
        
    def do_feedforward(self, inp):
        self.layer1 = self.sigmoid(np.dot(inp,self.weight1))
        self.output = self.layer1
        return output
        
    def backprop(self):
        
        self.w1_hold.append(self.weight1)
        out_der = self.sigmoid_der(self.output)


        
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1

        d_weights1 = np.dot(self.layer1.T, (2*(self.train_y - self.output) * out_der))

        self.weight1 += self.lr * d_weights1
   


# In[4]:


def linear_reg(feature, label):
    
    hold = np.linalg.inv(feature.T.dot(feature)).dot(feature.T)
    w = hold.dot(label)
    
    return w


# In[5]:


collector = []
with open('train1') as csv_file:    
    train = csv.reader(csv_file, delimiter=',')    
    for row in train:   
        collector.append(row)

x = 0.0
y = 0.0
train1 = np.zeros((collector.__len__(),2))

for i in range(collector.__len__()):
    a = collector[i][0]
    counter = 0

    for x in a:
        counter += 1
        if x == '\t':
            x_t = float(a[:counter])
            y_t = float(a[counter:])
    train1[i,0] = x_t
    train1[i,1] = y_t
    
#print(train1)

train_f = train1[:,0]
train_l = train1[:,1]


# In[6]:


collector = []
with open('test1') as csv_file:    
    train = csv.reader(csv_file, delimiter=',')    
    for row in train:   
        collector.append(row)


x = 0.0
y = 0.0
test1 = np.zeros((collector.__len__(),2))

for i in range(collector.__len__()):
    a = collector[i][0]
    counter = 0

    for x in a:
        counter += 1
        if x == '\t':
            x_t = float(a[:counter])
            y_t = float(a[counter:])
    test1[i,0] = x_t
    test1[i,1] = y_t
    
#print(test1)

test_f = test1[:,0]
test_l = test1[:,1]


# In[7]:


weight = linear_reg(train_f.reshape(60,1), train_l.reshape(60,1))


# In[8]:


print(weight)
print(weight.shape)


# In[9]:


est = test_f.reshape(41,1).dot(weight)

summer = 0

for i in range(test_l.shape[0]):
    
    summer = summer + (est[i] - test_l[i])**2/test_l.shape[0]
    
print("Sum of squared error in total by using linear regression is: " + str(summer))


# In[10]:


plt.plot(train_f, train_f.reshape(60,1).dot(weight))


# In[11]:


train_f_nor = (train_f -  np.min(train_f))/(np.max(train_f) - np.min(train_f))
train_l_nor = (train_l -  np.min(train_l))/(np.max(train_l) - np.min(train_l))

test_f_nor = (test_f -  np.min(test_f))/(np.max(test_f) - np.min(test_f))
test_l_nor = (test_f -  np.min(test_l))/(np.max(test_l) - np.min(test_l))


# # Question 2 A

# ## Finding the best model between ANN and Linear Regressor

# In[12]:


lr = 0.01
epoch = 80
hid_no = 32

nn = NN(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), hid_no, epoch, lr)

nn_test_rmse, nn_train_rmse, res_train, res_test = nn.train()


# In[13]:


lr = 0.01
epoch = 80
hid_no = 32

lr_nor = LinearReg(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), epoch, lr)

linear_test_rmse, linear_train_rmse = lr_nor.train()





# In[14]:


plt.plot(np.arange(80) + 1, nn_train_rmse)
plt.plot(np.arange(80) + 1, linear_train_rmse)
plt.legend(["ANN Training SSE", "Linear Regressor Training SSE"])
plt.title("ANN SSE Training vs. Linear Regressor SSE Training in 0.01 Learning Rate")
plt.ylabel("SSE Value")
plt.xlabel("Epoch Number")
plt.show()


# In[15]:


plt.plot(np.arange(80) + 1, nn_test_rmse)
plt.plot(np.arange(80) + 1, linear_test_rmse)
plt.legend(["ANN Test MSE", "Linear Regressor Test MSE"])
plt.title("ANN MSE Test vs. Linear Regressor MSE Test in 0.01 Learning Rate")
plt.ylabel("MSE Value")
plt.xlabel("Epoch Number")
plt.show()


# In[16]:


lr = 0.01
epoch = 80
hid_no = 32

nn_nor = NN(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), hid_no, epoch, lr)

nn_test_rmse_nor, nn_train_rmse_nor, res_train, res_test = nn_nor.train()


nn = NN(train_f.reshape(60,1), train_l.reshape(60,1), test_f.reshape(41,1), test_l_nor.reshape(41,1), hid_no, epoch, lr)

nn_test_rmse, nn_train_rmse, res_train, res_test = nn.train()


# ## Normalized Model vs. Non-Normalized Model

# In[17]:


plt.plot(np.arange(80) + 1, nn_train_rmse_nor)
plt.title("Normalized ANN Model Training SSE")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, nn_train_rmse)
plt.title("Non-Normalized ANN Model Training SSE")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, nn_test_rmse_nor)
plt.title("Normalized ANN Model Test SSE")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, nn_test_rmse)
plt.title("Non-Normalized ANN Model Test SSE")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()


# In[18]:


lr_arr = [10, 1 , 0.1, 0.01, 0.001, 0.0001]
epoch = 80
hid_no = 32
train_lr = []
test_lr = []

for i in lr_arr:
    nn_nor = NN(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), hid_no, epoch, i)
    nn_test_rmse_nor, nn_train_rmse_nor, res_train, res_test = nn_nor.train()
    train_lr.append(nn_train_rmse_nor)
    test_lr.append(nn_test_rmse_nor)
    


    


# In[19]:


train_lr = np.array(train_lr)
test_lr = np.array(test_lr)


# In[20]:


plt.plot(np.arange(80) + 1, train_lr[0])
plt.title("ANN Model Training SSE Learning Rate = 10")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, train_lr[1])
plt.title("ANN Model Training SSE Learning Rate = 1")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, train_lr[2])
plt.title("ANN Model Training SSE Learning Rate = 0.1")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, train_lr[3])
plt.title("ANN Model Training SSE Learning Rate = 0.01")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, train_lr[4])
plt.title("ANN Model Training SSE Learning Rate = 0.001")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, train_lr[5])
plt.title("ANN Model Training SSE Learning Rate = 0.0001")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()


# In[21]:


plt.plot(np.arange(80) + 1, test_lr[0])
plt.title("ANN Model Test SSE Learning Rate = 10")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, test_lr[1])
plt.title("ANN Model Test SSE Learning Rate = 1")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, test_lr[2])
plt.title("ANN Model Test SSE Learning Rate = 0.1")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, test_lr[3])
plt.title("ANN Model Test SSE Learning Rate = 0.01")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, test_lr[4])
plt.title("ANN Model Test SSE Learning Rate = 0.001")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()

plt.plot(np.arange(80) + 1, test_lr[5])
plt.title("ANN Model Test SSE Learning Rate = 0.0001")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()


# In[22]:


lr = 0.01
epoch = 700
hid_no = 32

nn_nor = NN(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), hid_no, epoch, lr)

nn_test_rmse_nor, nn_train_rmse_nor, res_train, res_test = nn_nor.train()


# In[23]:


plt.plot(np.arange(700), nn_train_rmse_nor)
plt.title("For finding the best epoch, used 700 epoch training")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()


plt.plot(np.arange(700), nn_test_rmse_nor)
plt.title("For finding the best epoch, used 700 epoch test")
plt.xlabel("Epoch Number")
plt.ylabel("SSE")
plt.show()


# ## Question 2 B

# In[24]:


lr = 0.01
epoch = 200
hid_no = 64

nn_nor = NN(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), hid_no, epoch, lr)

nn_test_rmse_nor, nn_train_rmse_nor, train_res, test_res = nn_nor.train()


# In[25]:


plt.plot(train_f_nor, train_l_nor)
plt.plot(train_f_nor, train_res)
plt.legend(["Real Values Graph", "Estimated Values Graph"])
plt.title("Real Values vs. Estimated Values in Training")
plt.ylabel("SSE Value")
plt.xlabel("Epoch Number")
plt.show()

plt.plot(test_f_nor, test_l_nor)
plt.plot(test_f_nor, test_res)
plt.legend(["Real Values Graph", "Estimated Values Graph"])
plt.title("Real Values vs. Estimated Values Test")
plt.ylabel("SSE Value")
plt.xlabel("Epoch Number")
plt.show()


# ## Question 2 C

# In[26]:


lr = 0.01
epoch = 1000
hid_no = [2, 4, 8, 16, 32, 64]
train_lr = []
test_lr = []

train_loss = []
test_loss = []

for i in hid_no:

    nn_nor = NN(train_f_nor.reshape(60,1), train_l_nor.reshape(60,1), test_f_nor.reshape(41,1), test_l_nor.reshape(41,1), i, epoch, lr)
    nn_test_rmse_nor, nn_train_rmse_nor, res_train, res_test = nn_nor.train()
    train_loss.append(nn_train_rmse_nor)
    test_loss.append(nn_test_rmse_nor)
    
    title = "Real Values vs. Estimated Values in Training For Hidden No " + str(i)
    plt.plot(train_f_nor, train_l_nor)
    plt.plot(train_f_nor, res_train)
    plt.legend(["Real Values Graph", "Estimated Values Graph"])
    plt.title(title)
    plt.ylabel("SSE Value")
    plt.xlabel("Epoch Number")
    plt.show()

train_loss = np.array(train_loss)
test_loss = np.array(test_loss)


# In[27]:


print(train_loss.shape)
print(test_loss.shape)


# In[28]:


mean_train = []
mean_test = []
std_train = []
std_test = []

for i in range(6):
    
    mean_train.append(np.mean(train_loss[i]))
    mean_test.append(np.mean(test_loss[i]))
    
    std_train.append(np.std(train_loss[i]))
    std_test.append(np.std(test_loss[i]))
    
    


# In[29]:


fig = go.Figure(data=[go.Table(header=dict(values=['Hidden No.', 'Mean Train', 'STD Train', 'Mean Test', 'STD Test']),
                 cells=dict(values=[[2, 4, 8, 16, 32, 64], mean_train, std_train, mean_test, std_test]))
                     ])
fig.show()

