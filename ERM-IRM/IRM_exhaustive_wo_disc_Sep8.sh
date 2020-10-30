


### Reproducing the results from the regression experiments in the manuscript 
### Before running this file please go to IRM_exhaustive_argparse_hper_wo_disc_Sep8.py and enter the path to the directory where you want to save the plots and store it in the variable we already have defined dir_name 


## dim: dimension of the data (n in the manuscript), n_reps 1 (keep it as 1), n_repits: number of repititoins, setup_hidden: 0- no confounders, 1-confounders present, setup_scramble: 0: scrambling matrix is identity, 1: S is an orthogonal transformation of the data (for our experiments in the manuscript we only worked with identity)  
## setup_hetero: 1- homoskedastic model (as the one described in the manuscript) and 0: heteroskedastic model (for our experiments we only worked with homoskedastic model), ones: 0 (keep it as 0), child: 0- no anticausal effects present, 1-anti-causal effects present


## reproduces CS-regression
python IRM_exhaustive_argparse_hper_wo_disc_Sep8.py --dim 10 --n_reps 1 --n_repits 25  --setup_hidden 0  --setup_hetero 1 --setup_scramble 0 --noise_identity 0 --ones 0 --child 0

## reproduces CF-regression
python IRM_exhaustive_argparse_hper_wo_disc_Sep8.py --dim 10 --n_reps 1 --n_repits 25  --setup_hidden 1  --setup_hetero 1 --setup_scramble 0 --noise_identity 0 --ones 0 --child 0 

## reproduces AC-regression
python IRM_exhaustive_argparse_hper_wo_disc_Sep8.py --dim 10 --n_reps 1 --n_repits 25  --setup_hidden 0  --setup_hetero 1 --setup_scramble 0 --noise_identity 0 --ones 0 --child 1

## reproduces HB-regression
python IRM_exhaustive_argparse_hper_wo_disc_Sep8.py --dim 10 --n_reps 1 --n_repits 25  --setup_hidden 1  --setup_hetero 1 --setup_scramble 0 --noise_identity 0 --ones 0 --child 1

