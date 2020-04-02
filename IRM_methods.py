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




class fixed_irm_game_model:
    def __init__(self, model_list, learning_rate, num_epochs, batch_size, termination_acc, warm_start):
        
        self.model_list        = model_list             # list of models for all the environments
        self.num_epochs        = num_epochs             # number of epochs 
        self.batch_size        = batch_size             # batch size for each gradient update
        self.termination_acc   = termination_acc        # threshold on accuracy below which we terminating 
        self.warm_start        = warm_start             # minimum number of steps we have to train before terminating due to accuracy falling below threshold
        self.learning_rate     = learning_rate          # learning rate in adam
    
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
            
        # cross entropy loss
        def loss_comb(model_list, x, y):
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            n_e = len(model_list)
            y_ = tf.zeros_like(y, dtype=tf.float32)
            # predict the model output from the ensemble
            for i in range(n_e):
                model_i = model_list[i]
                y_ = y_ + 0.5*model_i(x)

            return loss_object(y_true=y, y_pred=y_)
        # gradient of cross entropy loss for environment e
        def grad_comb(model_list, inputs, targets, e):
            with tf.GradientTape() as tape:
                loss_value = loss_comb(model_list, inputs, targets)
            return loss_value, tape.gradient(loss_value, model_list[e].trainable_variables)



    
        model_list = self.model_list
        learning_rate = self.learning_rate


        # initialize optimizer for all the environments and representation learner and store it in a list
        optimizer_list = []
        for e in range(n_e):
            optimizer_list.append(tf.keras.optimizers.Adam(learning_rate=learning_rate))

        ####### train

        train_accuracy_results_0 = []   # list to store training accuracy


        flag = 'false'
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        num_examples= data_tuple_list[0][0].shape[0]
        period      = n_e               
        termination_acc = self.termination_acc
        warm_start      = self.warm_start
        steps           = 0
        for epoch in range(num_epochs):
            print ("Epoch: "  + str(epoch))
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
                countp = count % period # countp decides the index of the model which trains in the current step
                for e in range(n_e):
                    batch_x_list.append(datat_list[e][0][offset:end,:])
                    batch_y_list.append(datat_list[e][1][offset:end,:])
                    loss_value, grads = grad_comb(model_list, batch_x_list[e], batch_y_list[e],e)
                    grads_list.append(grads)
                # update the environment whose turn it is to learn
                optimizer_list[countp].apply_gradients(zip(grads_list[countp], model_list[countp].trainable_variables))

                # computing training accuracy
                y_ = tf.zeros_like(y_in, dtype=tf.float32)
                for e in range(n_e):
                    y_ = y_ + model_list[e](x_in)
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                acc_train = np.float(epoch_accuracy(y_in, y_))
                train_accuracy_results_0.append(acc_train)
                
                if(steps>=warm_start and acc_train<termination_acc): ## Terminate after warm start and train acc touches threshold we dont want it to fall below 
                    flag = 'true' 
                    break      

                count = count +1
                steps = steps +1
                self.train_accuracy_results = train_accuracy_results_0
            if (flag == 'true'):
                break
        self.model_list = model_list 
        
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
        train_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()

        ytr_ = tf.zeros_like(y_in, dtype=tf.float32)
        for e in range(n_e):
            ytr_ = ytr_ + model_list[e](x_in)
        train_acc =  np.float(train_accuracy(y_in, ytr_))

        yts_ = tf.zeros_like(y_test, dtype=tf.float32)
        for e in range(n_e):
            yts_ = yts_ + model_list[e](x_test) 

        test_acc  =  np.float(test_accuracy(y_test, yts_))
        
        self.train_acc = train_acc
        self.test_acc  = test_acc

class variable_irm_game_model:
    def __init__(self, model_list, learning_rate, num_epochs, batch_size, termination_acc, warm_start):
        
        self.model_list        = model_list          # list of models for the environments and representation learner
        self.num_epochs        = num_epochs          # number of epochs
        self.batch_size        = batch_size          # batch size for each gradient update
        self.termination_acc   = termination_acc     # threshold on accuracy below which we terminate
        self.warm_start        = warm_start          # minimum number of steps before terminating
        self.learning_rate     = learning_rate       # learning rate for Adam optimizer
    
    def fit(self, data_tuple_list):
        n_e  = len(data_tuple_list)                                # number of environments
        # combine the data from the different environments x_in: combined data (features) from environments, y_in: combined labels from environments, e_in: combined environment indices from environments
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
            
        # cross entropy loss
        def loss_comb(model_list, x, y):
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            n_e = len(model_list)-1
            y_ = tf.zeros_like(y, dtype=tf.float32)
            # pass the data from the representation learner
            z = model_list[n_e](x) 
            # pass the output from the representation learner into the environments and aggregate them 
            for i in range(n_e):
                model_i = model_list[i]
                y_ = y_ + 0.5*model_i(z)


            return loss_object(y_true=y, y_pred=y_)

        # gradient of cross entropy loss for environment e
        def grad_comb(model_list, inputs, targets, e):
            with tf.GradientTape() as tape:
                loss_value = loss_comb(model_list, inputs, targets)
            return loss_value, tape.gradient(loss_value, model_list[e].trainable_variables)


        model_list = self.model_list
        learning_rate = self.learning_rate
        
        # initialize optimizer for all the environments and representation learner and store it in a list
        optimizer_list = []
        for e in range(n_e+1):
            if (e<=n_e-1):
                optimizer_list.append(tf.keras.optimizers.Adam(learning_rate=learning_rate))
            if (e==n_e):
                optimizer_list.append(tf.keras.optimizers.Adam(learning_rate=learning_rate*0.1))


        ####### train

        train_accuracy_results_0 = []   # list to store training accuracy  


        flag = 'false'
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        num_examples= data_tuple_list[0][0].shape[0]
        period      = n_e+1               # 
        termination_acc = self.termination_acc
        warm_start      = self.warm_start
        steps           = 0
        for epoch in range(num_epochs):
            print ("Epoch: " + str(epoch))
            datat_list = []
            for e in range(n_e+1):
                if(e<=n_e-1):
                    x_e = data_tuple_list[e][0]
                    y_e = data_tuple_list[e][1]
                    datat_list.append(shuffle(x_e,y_e)) 
                if (e==n_e):
                    datat_list.append(shuffle(x_in,y_in)) 
            count = 0
            for offset in range(0,num_examples, batch_size):
                end = offset + batch_size
                batch_x_list = [] # list to store batches for each environment
                batch_y_list = [] # list to store batches of labels for each environment
                loss_value_list = [] # list to store loss values
                grads_list      = [] # list to store gradients
                countp = period- 1- (count % period)  # countp decides the index of the model which trains in the current step
                for e in range(n_e+1):
                    batch_x_list.append(datat_list[e][0][offset:end,:])
                    batch_y_list.append(datat_list[e][1][offset:end,:])
                    loss_value, grads = grad_comb(model_list, batch_x_list[e], batch_y_list[e],e)
                    grads_list.append(grads)

                # update either a representation learner or an environment model  
                optimizer_list[countp].apply_gradients(zip(grads_list[countp], model_list[countp].trainable_variables))

                # computing training accuracy
                y_ = tf.zeros_like(y_in, dtype=tf.float32)
                z_in = model_list[n_e](x_in)
                for e in range(n_e):
                    y_ = y_ + model_list[e](z_in)
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                acc_train = np.float(epoch_accuracy(y_in, y_))
                train_accuracy_results_0.append(acc_train)
                
                if(steps>=warm_start and acc_train<termination_acc): # Terminate after warm start and train accuracy touches threshold we dont want it to fall below 
                    flag = 'true' 
                    break      
                count = count +1
                steps = steps +1
                self.train_accuracy_results = train_accuracy_results_0
            if (flag == 'true'):
                break
                
        self.model_list = model_list 
        
        self.x_in      = x_in
        self.y_in      = y_in

        
        
        
    def evaluate(self, data_tuple_test):
        ##### evaluations jmtd
        x_test = data_tuple_test[0]
        y_test = data_tuple_test[1]
        x_in   = self.x_in
        y_in   = self.y_in
        
        model_list = self.model_list
        n_e        = len(model_list)-1
        train_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        

        # compute training accuracy
        ytr_ = tf.zeros_like(y_in, dtype=tf.float32)
        z_in = model_list[n_e](x_in)
        for e in range(n_e):
            ytr_ = ytr_ + model_list[e](z_in)
        train_acc =  np.float(train_accuracy(y_in, ytr_))

        # compute testing accuracy
        z_test = model_list[n_e](x_test)
        yts_ = tf.zeros_like(y_test, dtype=tf.float32)
        for e in range(n_e):
            yts_ = yts_ + model_list[e](z_test) 

        test_acc  =  np.float(test_accuracy(y_test, yts_))
        
        self.train_acc = train_acc
        self.test_acc  = test_acc

class standard_erm_model:
    def __init__(self, model,  num_epochs, batch_size, learning_rate):
        
        self.model        = model
        self.num_epochs   = num_epochs
        self.batch_size   = batch_size
        self.learning_rate=learning_rate
    
    def fit(self, data_tuple_list):
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs
        n_e  = len(data_tuple_list)
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
        

    ### fit the model
        model = self.model
        batch_size = self.batch_size


        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

        model.fit(x_in, y_in,  epochs=num_epochs, batch_size=batch_size)
  
        self.x_in      = x_in
        self.y_in      = y_in      
        


        
        
        
    def evaluate(self, data_tuple_test):
        ##### evaluations jmtd
        x_test = data_tuple_test[0]
        y_test = data_tuple_test[1]
        x_in   = self.x_in
        y_in   = self.y_in
        
        model = self.model
        train_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()

        ytr_ = model.predict(x_in)
        train_acc =  np.float(train_accuracy(y_in, ytr_))

        yts_ = model.predict(x_test)

        test_acc  =  np.float(test_accuracy(y_test, yts_))
        
        self.train_acc = train_acc
        self.test_acc  = test_acc


class irm_model:
    def __init__(self, model, learning_rate, batch_size, steps_max, steps_threshold, gamma_new):
        self.model= model                                 # initialized model passed
        self.learning_rate = learning_rate                # learning rate for Adam optimizer
        self.batch_size    = batch_size                   # batch size per gradient update
        self.steps_max     = steps_max                    # maximum number of gradient steps
        self.steps_threshold = steps_threshold            # threshold on the number of steps after which we use penalty gamma_new
        self.gamma_new      = gamma_new                   # penalty value; note penalty is set to 1 initially and gamma_new only kicks in after steps exceeed steps_threshold
    
    def fit(self,data_tuple_list):
        
        n_e  = len(data_tuple_list)                       # number of environments
        # combine the data from the different environments, x_in: combined data (features) from different environments
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
        
        self.x_in = x_in
        self.y_in = y_in
        
        # cross entropy (we do not use the cross entropy from keras because there are issues when computing gradient of the gradient)
        def cross_entropy_manual(y,y_pred):
            y_p = tf.math.log(tf.nn.softmax(y_pred))
            n_p = np.float(tf.shape(y_p)[0])
            ind_0 = tf.where(y==0)[:,0]
            ind_1 = tf.where(y==1)[:,0]
            y_p0  = tf.gather(y_p, ind_0)[:,0]
            y_p1  = tf.gather(y_p, ind_1)[:,1]
            ent_0 = tf.reduce_sum(y_p0)
            ent_1 = tf.reduce_sum(y_p1)
            total = -(ent_0 + ent_1)/n_p
            return total

        # cross entropy loss for environment e
        def loss_n(model,x,e,y,w,k):
            index = np.where(e==k)
            y1_ = model(x[index[0]])*w
            y1  = y[index[0]]

            return cross_entropy_manual(y1,y1_)   

        # gradient of cross entropy loss w.r.t w for environment e
        def grad_norm_n(model,x,e,y,w,k):
            with tf.GradientTape() as g:
                g.watch(w)
                loss_value = loss_n(model,x,e,y,w,k)  
            return g.gradient(loss_value, w)**2

        # total cross entropy loss across all environments    
        def loss_0(model, x,e,y,w):
            y_ =model(x)
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            return loss_object(y_true=y, y_pred=y_)
        
        # sum of cross entropy loss and penalty 
        def loss_total(model,x,e,y,w,gamma, n_e):
            loss0 = loss_0(model,x,e,y,w)
            loss_penalty = 0.0
            for k in range(n_e):
                loss_penalty += gamma*grad_norm_n(model,x,e,y,w,k)

            return (loss0 + loss_penalty)*(1/gamma)        
        
        # gradient of sum of cross entropy loss and penalty w.r.t model parameters
        def grad_total_n(model,x,e,y,w,gamma, n_e):
            with tf.GradientTape() as tape:
                loss_value = loss_total(model,x,e,y,w,gamma,n_e)
            return loss_value, tape.gradient(loss_value, model.trainable_variables)
        
        model = self.model
        learning_rate = self.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        ## train 
        train_loss_results = []
        train_accuracy_results = []
        flag = 'false'
        batch_size = self.batch_size
        num_examples= x_in.shape[0]
        gamma     = 1.0
        w         = tf.constant(1.0)
        steps     = 0
        steps_max  = self.steps_max
        steps_threshold = self.steps_threshold
        gamma_new       = self.gamma_new
        while(steps<=steps_max):
            (xt,yt,et)   = shuffle(x_in,y_in,e_in)
            epoch_loss_avg = tf.keras.metrics.Mean()
            epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
            count = 0
            if(steps>=steps_threshold):
                gamma = gamma_new
            for offset in range(0,num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y, batch_e = xt[offset:end,:], yt[offset:end,:], et[offset:end,:]
                loss_values, grads = grad_total_n(model,batch_x,batch_e,batch_y,w,gamma,n_e)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss_avg(loss_values)    
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                acc_train = np.float(epoch_accuracy(y_in, model(x_in)))
                train_loss_results.append(epoch_loss_avg.result())
                train_accuracy_results.append(epoch_accuracy.result())
                count = count +1 
                steps = steps + 1
            
    def evaluate(self, data_tuple_test):
        x_test = data_tuple_test[0]
        y_test = data_tuple_test[1]
        x_in   = self.x_in
        y_in   = self.y_in
        train_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        model  = self.model
        ytr_ = model.predict(x_in)
        train_acc =  np.float(train_accuracy(y_in, ytr_))
        
        yts_ = model.predict(x_test)
        test_acc =  np.float(test_accuracy(y_test, yts_))
        
        self.train_acc = train_acc
        self.test_acc  = test_acc
