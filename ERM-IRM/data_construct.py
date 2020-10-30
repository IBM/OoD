import tensorflow as tf
import numpy as np
import argparse
import IPython.display as display
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pandas as pd
tf.compat.v1.enable_eager_execution()
import cProfile
from sklearn.model_selection import train_test_split
import copy as cp
from sklearn.model_selection import KFold
from datetime import date
import time

# Data generation for CS-CMNIST
class assemble_data_mnist_sb:
    def __init__(self, n_tr):
        
        D=tf.keras.datasets.mnist.load_data()
        n_tr_total = D[0][0].shape[0]
        ind_tr =np.random.choice(n_tr_total, n_tr)
        x_train=D[0][0][ind_tr].astype('float32')
        x_test=D[1][0].astype('float32')
        num_train=x_train.shape[0]
        self.x_train_mnist=x_train.reshape((num_train,28,28,1))
        self.y_train_mnist=D[0][1][ind_tr].reshape((num_train,1))
        num_test=x_test.shape[0]
        self.x_test_mnist=x_test.reshape((num_test,28,28,1))
        self.y_test_mnist=D[1][1].reshape((num_test,1))
        self.n_tr = n_tr

    def create_environment(self,env_index,x,y,prob_e,prob_label): 
        # prob_label we retain from other classes for simplicity but is not relevant for this class
        
        #Convert y>5 to 1 and y<5 to 0.
        y= (y>=5).astype(int) # binarize the digit labels
        num_samples=len(y)
        
        z_color = np.random.binomial(1,0.5,(num_samples,1)) # sample color for each sample
        w_comb  =1- np.logical_xor(y, z_color)              # compute xor of label and color and negate it

        selection_0 = np.where(w_comb==0)[0]                # indices where -xor is zero
        selection_1 = np.where(w_comb==1)[0]                # indices were -xor is one
        
        ns0 = np.shape(selection_0)[0]
        ns1 = np.shape(selection_1)[0]
        
        final_selection_0 = selection_0[np.where(np.random.binomial(1,prob_e,(ns0,1))==1)[0]]   # -xor =0 then select that point with probability prob_e
        final_selection_1 = selection_1[np.where(np.random.binomial(1,1-prob_e,(ns1,1))==1)[0]] # -xor =0 then select that point with probability 1-prob_e

        final_selection = np.concatenate((final_selection_0, final_selection_1), axis=0)        # indices of the final set of points selected
        
        
        z_color_final = z_color[final_selection]  # colors of the final set of selected points
        y=y[final_selection]                      # labels of the final set of selected points
        x= x[final_selection]                     # gray scale image of the final set of selected points
        
        ### color the points x based on z_color_final 
        
        red = np.where(z_color_final==0)[0]       # select the points with z_color_final=0 to set them to red color
        green = np.where(z_color_final==1)[0]     # select the points with z_color_final=1 to set them to green color
        
        num_samples_final = np.shape(y)[0]

        tsh = 0.5
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG, chB), axis=3)

        tsh= 0.5
        chR1= cp.deepcopy(x[green,:])
        chR1[chR1 > tsh] = 0
        chG1= cp.deepcopy(x[green,:])
        chG1[chG1 > tsh] = 1
        chB1= cp.deepcopy(x[green,:])
        chB1[chB1 > tsh] = 0
        g= np.concatenate((chR1, chG1, chB1), axis=3)


        dataset=np.concatenate((r,g),axis=0)
        labels=np.concatenate((y[red,:],y[green,:]),axis=0)

        return (dataset,labels,np.ones((num_samples_final,1))*env_index, z_color_final)
    
    

    
    def create_training_data(self, n_e, corr_list, p_label_list):
        x_train_mnist = self.x_train_mnist
        y_train_mnist = self.y_train_mnist
        n_tr = self.n_tr
        ind_X = range(0,n_tr)
        kf = KFold(n_splits=n_e, shuffle=True)
        l=0
        ind_list =[]
        for train, test in kf.split(ind_X):
            ind_list.append(test)
            l=l+1   
        data_tuple_list = []
        for l in range(n_e):
            data_tuple_list.append(self.create_environment(l,x_train_mnist[ind_list[l],:,:,:],y_train_mnist[ind_list[l],:],corr_list[l],p_label_list[l]))

        self.data_tuple_list = data_tuple_list
            
    def create_testing_data(self, corr_test, prob_label, n_e):
        x_test_mnist = self.x_test_mnist
        y_test_mnist = self.y_test_mnist        
        (x_test,y_test,e_test,z_color_test)=self.create_environment(n_e,x_test_mnist,y_test_mnist,corr_test,prob_label)

        self.data_tuple_test = (x_test,y_test, e_test)
        
        
    def create_testing_data_blue(self,  n_e):
        x_test_mnist = self.x_test_mnist
        y_test_mnist = self.y_test_mnist        
        (x_test,y_test,e_test)=self.create_environment_blue(n_e,x_test_mnist,y_test_mnist)

        self.data_tuple_test_blue = (x_test,y_test, e_test)

# Data generation for CF-CMNIST
class assemble_data_mnist_confounded:
    def __init__(self,n_tr):
        D=tf.keras.datasets.mnist.load_data()
        n_tr_total = D[0][0].shape[0]
        print(n_tr_total)
        ind_tr =np.random.choice(n_tr_total, n_tr)
        x_train=D[0][0][ind_tr].astype(float)
        #y_train=OneHotEncoder.fit_transform(y_train)
        x_test=D[1][0].astype(float)
        #y_test=OneHotEncoder.fit_transform(y_train)
        num_train=x_train.shape[0]
        self.x_train_mnist=x_train.reshape((num_train,28,28,1))
        self.y_train_mnist=D[0][1][ind_tr].reshape((num_train,1))
        num_test=x_test.shape[0]
        self.x_test_mnist=x_test.reshape((num_test,28,28,1))
        self.y_test_mnist=D[1][1].reshape((num_test,1))
        self.n_tr = n_tr

    def create_environment(self,env_index,x,y,prob_e,prob_label):
        #Convert y>5 to 1 and y<5 to 0.
        y= (y>=5).astype(int)
        num_samples=len(y)
        h = np.random.binomial(1,prob_label,(num_samples,1))
        h1 = np.random.binomial(1,prob_e,(num_samples,1))
        y_mod=np.abs(y-h)    
        z=np.logical_xor(h1, h)



        red = np.where(z==1)[0]
        print(red.shape)
        tsh = 0.0
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG), axis=3)

        green= np.where(z==0)[0]
        print(green.shape)
        tsh= 0.0
        chR1= cp.deepcopy(x[green,:])
        chR1[chR1 > tsh] = 0
        chG1= cp.deepcopy(x[green,:])
        chG1[chG1 > tsh] = 1
        chB1= cp.deepcopy(x[green,:])
        chB1[chB1 > tsh] = 0
        g= np.concatenate((chR1, chG1), axis=3)


        dataset=np.concatenate((r,g),axis=0)
        labels=np.concatenate((y_mod[red,:],y_mod[green,:]),axis=0)

        return (dataset,labels,np.ones((num_samples,1))*env_index)
    
    def create_training_data(self, n_e, corr_list, p_label_list):
            x_train_mnist = self.x_train_mnist
            y_train_mnist = self.y_train_mnist
            n_tr = self.n_tr
            ind_X = range(0,n_tr)
            kf = KFold(n_splits=n_e, shuffle=True)
            l=0
            ind_list =[]
            for train, test in kf.split(ind_X):
                ind_list.append(test)
                l=l+1   
            data_tuple_list = []
            for l in range(n_e):
                data_tuple_list.append(self.create_environment(l,x_train_mnist[ind_list[l],:,:,:],y_train_mnist[ind_list[l],:],corr_list[l],p_label_list[l]))

            self.data_tuple_list = data_tuple_list
            
    def create_testing_data(self, corr_test, prob_label, n_e):
        x_test_mnist = self.x_test_mnist
        y_test_mnist = self.y_test_mnist        
        (x_test,y_test,e_test)=self.create_environment(n_e,x_test_mnist,y_test_mnist,corr_test,prob_label)

        self.data_tuple_test = (x_test,y_test, e_test)
# Data generation for CHD-CMNIST
class assemble_data_mnist_child:
    def __init__(self,n_tr):
        D=tf.keras.datasets.mnist.load_data()
        n_tr_total = D[0][0].shape[0]
        print(n_tr_total)
        ind_tr =np.random.choice(n_tr_total, n_tr)
        x_train=D[0][0][ind_tr].astype(float)
        #y_train=OneHotEncoder.fit_transform(y_train)
        x_test=D[1][0].astype(float)
        #y_test=OneHotEncoder.fit_transform(y_train)
        num_train=x_train.shape[0]
        self.x_train_mnist=x_train.reshape((num_train,28,28,1))
        self.y_train_mnist=D[0][1][ind_tr].reshape((num_train,1))
        num_test=x_test.shape[0]
        self.x_test_mnist=x_test.reshape((num_test,28,28,1))
        self.y_test_mnist=D[1][1].reshape((num_test,1))
        self.n_tr = n_tr

    def create_environment(self,env_index,x,y,prob_e,prob_label):
        #Convert y>5 to 1 and y<5 to 0.
        y= (y>=5).astype(int)
        num_samples=len(y)
        y_mod=np.abs(y-np.random.binomial(1,prob_label,(num_samples,1)))
        z=np.abs(y_mod-np.random.binomial(1,prob_e,(num_samples,1)))



        red = np.where(z==1)[0]
        tsh = 0.0
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG), axis=3)

        green= np.where(z==0)[0]
        tsh= 0.0
        chR1= cp.deepcopy(x[green,:])
        chR1[chR1 > tsh] = 0
        chG1= cp.deepcopy(x[green,:])
        chG1[chG1 > tsh] = 1
        chB1= cp.deepcopy(x[green,:])
        chB1[chB1 > tsh] = 0
        g= np.concatenate((chR1, chG1), axis=3)


        dataset=np.concatenate((r,g),axis=0)
        labels=np.concatenate((y_mod[red,:],y_mod[green,:]),axis=0)

        return (dataset,labels,np.ones((num_samples,1))*env_index)
    
    def create_training_data(self, n_e, corr_list, p_label_list):
            x_train_mnist = self.x_train_mnist
            y_train_mnist = self.y_train_mnist
            n_tr = self.n_tr
            ind_X = range(0,n_tr)
            kf = KFold(n_splits=n_e, shuffle=True)
            l=0
            ind_list =[]
            for train, test in kf.split(ind_X):
                ind_list.append(test)
                l=l+1   
            data_tuple_list = []
            for l in range(n_e):
                data_tuple_list.append(self.create_environment(l,x_train_mnist[ind_list[l],:,:,:],y_train_mnist[ind_list[l],:],corr_list[l],p_label_list[l]))

            self.data_tuple_list = data_tuple_list
            
    def create_testing_data(self, corr_test, prob_label, n_e):
        x_test_mnist = self.x_test_mnist
        y_test_mnist = self.y_test_mnist        
        (x_test,y_test,e_test)=self.create_environment(n_e,x_test_mnist,y_test_mnist,corr_test,prob_label)

        self.data_tuple_test = (x_test,y_test, e_test)


# Data generation for HB-CMNIST
class assemble_data_mnist_confounded_child:
    def __init__(self,n_tr):
        D=tf.keras.datasets.mnist.load_data()
        n_tr_total = D[0][0].shape[0]
        print(n_tr_total)
        ind_tr =np.random.choice(n_tr_total, n_tr)
        x_train=D[0][0][ind_tr].astype(float)
        #y_train=OneHotEncoder.fit_transform(y_train)
        x_test=D[1][0].astype(float)
        #y_test=OneHotEncoder.fit_transform(y_train)
        num_train=x_train.shape[0]
        self.x_train_mnist=x_train.reshape((num_train,28,28,1))
        self.y_train_mnist=D[0][1][ind_tr].reshape((num_train,1))
        num_test=x_test.shape[0]
        self.x_test_mnist=x_test.reshape((num_test,28,28,1))
        self.y_test_mnist=D[1][1].reshape((num_test,1))
        self.n_tr = n_tr

    def create_environment(self,env_index,x,y,prob_e,prob_label):
        #Convert y>5 to 1 and y<5 to 0.
        y= (y>=5).astype(int)
        num_samples=len(y)
        h = np.random.binomial(1,prob_label,(num_samples,1))
        
        h1 = np.random.binomial(1,prob_e,(num_samples,1))
        w  = np.random.binomial(1,0.8,(num_samples,1))
        y_mod=np.abs(y-h)
        z = np.multiply(w,np.abs(y_mod-h1)) + np.multiply((1-w), np.abs(h-h1))

        

        red = np.where(z==1)[0]
        print(red.shape)
        tsh = 0.0
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG), axis=3)

        green= np.where(z==0)[0]
        print(green.shape)
        tsh= 0.0
        chR1= cp.deepcopy(x[green,:])
        chR1[chR1 > tsh] = 0
        chG1= cp.deepcopy(x[green,:])
        chG1[chG1 > tsh] = 1
        chB1= cp.deepcopy(x[green,:])
        chB1[chB1 > tsh] = 0
        g= np.concatenate((chR1, chG1), axis=3)


        dataset=np.concatenate((r,g),axis=0)
        labels=np.concatenate((y_mod[red,:],y_mod[green,:]),axis=0)

        return (dataset,labels,np.ones((num_samples,1))*env_index)

    def create_environment1(self,env_index,x,y,prob_e,prob_label):
        #Convert y>5 to 1 and y<5 to 0.
        y= (y>=5).astype(int)
        num_samples=len(y)
        h = np.random.binomial(1,prob_label,(num_samples,1))
        h1 = np.random.binomial(1,prob_e,(num_samples,1))
        y_mod=np.abs(y-h)
#         y1= np.abs(y-h1)
        y1 = np.multiply(y,h1)
#         y2= 1-np.logical_xor(y1,y_mod) 
#         z=np.logical_xor(y2, h)
#         h2 = np.logical_xor(h1, h)
        z= 1-np.logical_xor(y_mod,y1)
#         print (z[:3])
        

        red = np.where(z==1)[0]
        print(red.shape)
        tsh = 0.0
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG), axis=3)

        green= np.where(z==0)[0]
        print(green.shape)
        tsh= 0.0
        chR1= cp.deepcopy(x[green,:])
        chR1[chR1 > tsh] = 0
        chG1= cp.deepcopy(x[green,:])
        chG1[chG1 > tsh] = 1
        chB1= cp.deepcopy(x[green,:])
        chB1[chB1 > tsh] = 0
        g= np.concatenate((chR1, chG1), axis=3)


        dataset=np.concatenate((r,g),axis=0)
        labels=np.concatenate((y_mod[red,:],y_mod[green,:]),axis=0)

        return (dataset,labels,np.ones((num_samples,1))*env_index)
    
    def create_training_data(self, n_e, corr_list, p_label_list):
            x_train_mnist = self.x_train_mnist
            y_train_mnist = self.y_train_mnist
            n_tr = self.n_tr
            ind_X = range(0,n_tr)
            kf = KFold(n_splits=n_e, shuffle=True)
            l=0
            ind_list =[]
            for train, test in kf.split(ind_X):
                ind_list.append(test)
                l=l+1   
            data_tuple_list = []
            for l in range(n_e):
                data_tuple_list.append(self.create_environment(l,x_train_mnist[ind_list[l],:,:,:],y_train_mnist[ind_list[l],:],corr_list[l],p_label_list[l]))

            self.data_tuple_list = data_tuple_list
            
    def create_testing_data(self, corr_test, prob_label, n_e):
        x_test_mnist = self.x_test_mnist
        y_test_mnist = self.y_test_mnist        
        (x_test,y_test,e_test)=self.create_environment(n_e,x_test_mnist,y_test_mnist,corr_test,prob_label)

        self.data_tuple_test = (x_test,y_test, e_test)