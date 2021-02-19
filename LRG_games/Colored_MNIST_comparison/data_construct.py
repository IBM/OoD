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
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# import cProfile
import copy as cp
from sklearn.model_selection import KFold
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
        
class assemble_data_mnist_fashion:
    def __init__(self):

        fashion_mnist = keras.datasets.fashion_mnist ## Load fashion MNIST data
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


        ## Prepare the data with labels (footwear or a clothing item, drop label 8 as it is a bag)
        sub_labels_train = np.where(train_labels!=8)
        train_images_new = train_images[sub_labels_train]
        train_labels_new = train_labels[sub_labels_train]

        sub_labels_test = np.where(test_labels!=8)
        test_images_new = test_images[sub_labels_test]
        test_labels_new = test_labels[sub_labels_test]


        train_labels_binary  = np.zeros_like(train_labels_new)
        test_labels_binary  = np.zeros_like(test_labels_new)
        train_labels_binary[np.where(train_labels_new==5)] = 1
        train_labels_binary[np.where(train_labels_new==7)] = 1
        train_labels_binary[np.where(train_labels_new==9)] = 1
        test_labels_binary[np.where(test_labels_new==5)] = 1
        test_labels_binary[np.where(test_labels_new==7)] = 1
        test_labels_binary[np.where(test_labels_new==9)] = 1

        x_train_fashion_mnist = train_images_new
        x_test_fashion_mnist  = test_images_new
        y_train_fashion_mnist = train_labels_binary
        y_test_fashion_mnist  = test_labels_binary

        ## convert data into float b/w 0-1
        x_train_fashion_mnist=x_train_fashion_mnist.astype('float32')/float(255)
        x_test_fashion_mnist=x_test_fashion_mnist.astype('float32')/float(255)

        num_train = x_train_fashion_mnist.shape[0]
        x_train_fashion_mnist = x_train_fashion_mnist.reshape((num_train,28,28,1))
        y_train_fashion_mnist = y_train_fashion_mnist.reshape((num_train,1))
        
        
        
        num_test = x_test_fashion_mnist.shape[0]
        x_test_fashion_mnist = x_test_fashion_mnist.reshape((num_test,28,28,1))
        y_test_fashion_mnist = y_test_fashion_mnist.reshape((num_test,1))
        
        self.x_train_fashion_mnist = x_train_fashion_mnist
        self.y_train_fashion_mnist = y_train_fashion_mnist

        self.x_test_fashion_mnist = x_test_fashion_mnist
        self.y_test_fashion_mnist = y_test_fashion_mnist 
        
    def create_environment(self,env_index,x,y,prob_e,prob_label):
        y = y.astype(int)
        num_samples=len(y)   
        y_mod=np.abs(y-np.random.binomial(1,prob_label,(num_samples,1)))
        z=np.abs(y_mod-np.random.binomial(1,prob_e,(num_samples,1)))
        red = np.where(z==1)[0]
        tsh = 0.5
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG), axis=3)
        green= np.where(z==0)[0]
        tsh= 0.5
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
        x_train_fashion_mnist = self.x_train_fashion_mnist
        y_train_fashion_mnist = self.y_train_fashion_mnist
        ind_X = range(0,54000)
        kf = KFold(n_splits=n_e, shuffle=True)
        l=0
        ind_list =[]
        for train, test in kf.split(ind_X):
            ind_list.append(test)
            l=l+1   
        data_tuple_list = []
        for l in range(n_e):
            data_tuple_list.append(self.create_environment(l,x_train_fashion_mnist[ind_list[l],:,:,:],y_train_fashion_mnist[ind_list[l],:],corr_list[l],p_label_list[l]))
            
        self.data_tuple_list = data_tuple_list
        
    
    def create_testing_data(self, corr_test, prob_label, n_e):
        x_test_fashion_mnist = self.x_test_fashion_mnist
        y_test_fashion_mnist = self.y_test_fashion_mnist        
        (x_test,y_test,e_test)=self.create_environment(n_e,x_test_fashion_mnist,y_test_fashion_mnist,corr_test,prob_label)

        self.data_tuple_test = (x_test,y_test, e_test)

        
        
class assemble_data_mnist:
    def __init__(self):
        D=tf.keras.datasets.mnist.load_data()
        x_train=D[0][0].astype(float)
        #y_train=OneHotEncoder.fit_transform(y_train)
        x_test=D[1][0].astype(float)
        #y_test=OneHotEncoder.fit_transform(y_train)
        num_train=x_train.shape[0]
        self.x_train_mnist=x_train.reshape((num_train,28,28,1))
        self.y_train_mnist=D[0][1].reshape((num_train,1))
        num_test=x_test.shape[0]
        self.x_test_mnist=x_test.reshape((num_test,28,28,1))
        self.y_test_mnist=D[1][1].reshape((num_test,1))

    def create_environment(self,env_index,x,y,prob_e,prob_label):
        #Convert y>5 to 1 and y<5 to 0.
        y= (y>=5).astype(int)
        num_samples=len(y)
        y_mod=np.abs(y-np.random.binomial(1,prob_label,(num_samples,1)))
        z=np.abs(y_mod-np.random.binomial(1,prob_e,(num_samples,1)))



        red = np.where(z==1)[0]
        tsh = 0.5
        chR = cp.deepcopy(x[red,:])
        chR[chR > tsh] = 1
        chG = cp.deepcopy(x[red,:])
        chG[chG > tsh] = 0
        chB = cp.deepcopy(x[red,:])
        chB[chB > tsh] = 0
        r = np.concatenate((chR, chG), axis=3)

        green= np.where(z==0)[0]
        tsh= 0.5
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
            ind_X = range(0,60000)
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


