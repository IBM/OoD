
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
import argparse
import torch
import numpy

class fixed_irm_game_model_regression:
    def __init__(self, model_list, learning_rate, num_epochs, batch_size, bound, plot_flag):
        
        self.model_list        = model_list             # list of models for all the environments
        self.num_epochs        = num_epochs             # number of epochs 
        self.batch_size        = batch_size             # batch size for each gradient update
        self.learning_rate     = learning_rate
        self.lb                = -bound
        self.ub                =  bound
        self.plot_flag         =  plot_flag
    def fit(self, data_tuple_list):
        n_e  = len(data_tuple_list)                     # number of environments
        # combine the data from the different environments x_in: combined data from environments, y_in: combined labels from environments, e_in: combined environment indices from environments
        x_in = data_tuple_list[0][0]
        for i in range(1,n_e):
            x_c = data_tuple_list[i][0]
            x_in = np.concatenate((x_in, x_c), axis=0)
        y_in = data_tuple_list[0][1]
        for i in range(1,n_e):
            y_c = data_tuple_list[i][1]
            y_in = np.concatenate((y_in, y_c), axis=0)
        e_in = data_tuple_list[0][2]
        for i in range(1,n_e):
            e_c = data_tuple_list[i][2]
            e_in = np.concatenate((e_in, e_c), axis=0) 
            
        # MSE loss
        def loss_comb(model_list, x, y):
            loss_object = tf.keras.losses.MeanSquaredError()
            n_e = len(model_list)
            y_ = tf.zeros_like(y, dtype=tf.float64)
            # predict the model output from the ensemble
            for i in range(n_e):
                model_i = model_list[i]
                y_ = y_ + model_i(x)

            return loss_object(y_true=y, y_pred=y_)
        

        # gradient of MSE loss for environment e
        def grad_comb(model_list, inputs, targets, e):
            with tf.GradientTape() as tape:
                loss_value = loss_comb(model_list, inputs, targets)
            return loss_value, tape.gradient(loss_value, model_list[e].trainable_variables)

        
        def project_model(model, lb, ub):
            weights = model.weights[0]
            weights_array = weights.numpy()
            weights_array[weights_array<lb] = lb
            weights_array[weights_array>ub] = ub
            return weights_array
    
        model_list = self.model_list
        learning_rate = self.learning_rate


        # initialize optimizer for all the environments and representation learner and store it in a list
        optimizer_list = []
        for e in range(n_e):
            optimizer_list.append(tf.keras.optimizers.SGD(learning_rate=learning_rate))

        ####### train

        train_err_results_0 = []   # list to store training accuracy
        train_err_results_e0 = []
        train_err_results_e1 = []
        train_err_results = [[0.0]]*n_e 

        plot_flag = self.plot_flag
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        num_examples= data_tuple_list[0][0].shape[0]
        period      = n_e  
        P           = 2
        period_div  = P//n_e

        steps           = 0
        lb = self.lb
        ub = self.ub
        w1_1= []
        w1_2= []
        w2_1= []
        w2_2= []
        flag=0
        for epoch in range(num_epochs):
#             print ("Epoch: "  + str(epoch))
            datat_list = []
            for e in range(n_e):
                x_e = data_tuple_list[e][0]
                y_e = data_tuple_list[e][1]
                datat_list.append(shuffle(x_e,y_e)) 
            count = 0
            for offset in range(0,num_examples, batch_size):
                end = offset + batch_size
                batch_x_list = []  # list to store batches for each environment
                batch_y_list = []  # list to store batches of labels for each environment
                loss_value_list = []  # list to store loss values
                grads_list      = []  # list to store gradients
                count_rem = count % P # countp decides the index of the model which trains in the current step
                countp    = count_rem//period_div
                for e in range(n_e):
                    batch_x_list.append(datat_list[e][0][offset:end,:])
                    batch_y_list.append(datat_list[e][1][offset:end,:])
                    loss_value, grads = grad_comb(model_list, batch_x_list[e], batch_y_list[e],e)
                    grads_list.append(grads)
                optimizer_list[countp].apply_gradients(zip(grads_list[countp], model_list[countp].trainable_variables))
                w_array = project_model(model_list[countp], lb,ub)
                model_list[countp].weights[0].assign(w_array)

                # computing training error
                y_ = tf.zeros_like(y_in, dtype=tf.float64)
                for e in range(n_e):
                    y_ = y_ + model_list[e](x_in)
                epoch_error = tf.keras.metrics.MeanSquaredError()
                err_train = np.float(epoch_error(y_in, y_))
                train_err_results_0.append(err_train)

                for k in range(n_e):
                    x_c = data_tuple_list[k][0]
                    y_c = data_tuple_list[k][1]
                    y_ = tf.zeros_like(y_c, dtype=tf.float64)
                    for e in range(n_e):
                        y_ = y_ + model_list[e](x_c)
                    epoch_error = tf.keras.metrics.MeanSquaredError()
                    err_train = np.float(epoch_error(y_c, y_))
                    if(k==0):
                        train_err_results_e0.append(err_train)
                    if(k==1):
                        train_err_results_e1.append(err_train)       

                count = count +1
                steps = steps +1
                self.train_error_results = train_err_results_0
                self.train_error_results_env0 = train_err_results_e0
                self.train_error_results_env1 = train_err_results_e1
            if (plot_flag == 'true'):
                w1_1.append(model_list[0].weights[0].numpy().T[0][0])
                w1_2.append(model_list[0].weights[0].numpy().T[0][1])
                w2_1.append(model_list[1].weights[0].numpy().T[0][0])
                w2_2.append(model_list[1].weights[0].numpy().T[0][1])


        self.model_list = model_list
        self.w1_1 = w1_1
        self.w1_2 = w1_2
        self.w2_1 = w2_1
        self.w2_2 = w2_2    
        self.x_in      = x_in
        self.y_in      = y_in

        
        
    def evaluate(self, data_tuple_test):
        ##### evaluations jmtd
        x_test = data_tuple_test[0]
        y_test = data_tuple_test[1]
        x_in   = self.x_in
        y_in   = self.y_in
        
        model_list = self.model_list
        n_e        = len(model_list)
        train_error= tf.keras.metrics.MeanSquaredError()
        test_error= tf.keras.metrics.MeanSquaredError()

        ytr_ = tf.zeros_like(y_in, dtype=tf.float64)
        for e in range(n_e):
            ytr_ = ytr_ + model_list[e](x_in)
        train_err =  np.float(train_error(y_in, ytr_))

        yts_ = tf.zeros_like(y_test, dtype=tf.float64)
        for e in range(n_e):
            yts_ = yts_ + model_list[e](x_test) 

        test_err  =  np.float(test_error(y_test, yts_))
        
        self.train_err = train_err
        self.test_err  = test_err

def convert_regn_np_format(all_environments):
    n_e = len(all_environments[0])
    data_tuple_list = []
    for i in range(n_e):
        x_i = all_environments[0][i][0].numpy().astype('float')
        y_i = all_environments[0][i][1].numpy().astype('float')
        n_s = np.shape(x_i)[0]
        e_i = np.ones((n_s,1))*i
        data_tuple_list.append((x_i,y_i,e_i))
    return data_tuple_list