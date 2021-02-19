# IRM games

This library includes three files and a folder. 



1. IRM_games_jmtd_illstration_notebook.ipynb: Jupyter notebook serves as the tutorial. In this tutorial, we compare the IRM games methods, IRM from Arjovsky et.al. and standard ERM. 


2. data_construct.py: we define two classes in this file; one class corresponds to colored MNIST digits and other class corresponds to colored Fashion MNIST. These classes give the functionality of how much correlation we want to induce with the colors, how many environments we want. (See the tutorial notebook for details)


3. IRM_methods.py: consists of four classes. 


    a. fixed_irm_game_model
    
    b. variable_irm_game_model
    
    c. irm_model
    
    d. standard_erm_model

    Each of these classes have attributes similar to any sklearn class. Initialization requires us to specify the hyper-parameters for the class. fit function is called for training and evaluate function is used for evaluation. (See tutorial notebook for details).
 



## Suggested Citation

Kartik Ahuja, Karthikeyan Shanmugam, Kush R. Varshney, and Amit Dhurandhar, "Invariant Risk Minimization Games," arXiv:2002.04692, 2020.

    @booklet{AhujaSVD2020,
        author="Kartik Ahuja and Karthikeyan Shanmugam and Kush R. Varshney and Amit Dhurandhar",
        title="Invariant Risk Minimization Games",
        howpublished="International Conference on Machine Learning",
        year="2020"
    }

