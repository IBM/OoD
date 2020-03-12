# IRM games

This library includes three files. 

IRM_games_jmtd_illstration_notebook.ipynb: Jupyter notebook serves as the tutorial where we compare the IRM games methods, IRM from Arjovsky et.al. and standard ERM. 


data_construct.py: we define two classes in this file; one class corresponds to colored MNIST digits and other class corresponds to colored Fashion MNIST. These classes give the functionality of how much correlation we want to induce with the colors, how many environments we want. (See the tutorial notebook for details)


IRM_methods.py: consists of four classes. 


1. fixed_irm_game_model
2. variable_irm_game_model
3. irm_model
4. standard_erm_model

Each of these classes have attributes similar to any sklearn class. Initialization requires us to specify the hyper-parameters for the class. fit function is called for training and evaluate function is used for evaluation. (See tutorial notebook for details).

