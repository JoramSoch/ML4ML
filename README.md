# ML4ML

### Machine Learning Analysis for MATLAB

_A MATLAB Wrapper for Machine Learning, with another wrapper for neuroimaging data analysis_


## Requirements

These functions require the following packages:

* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (for `ML_SVC` and `ML_SVR`) for MATLAB
* [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) (for `ML4NI_SVM` and `ML4NI_mask`), preferably [SPM12](https://github.com/spm/spm12).


## Getting Started

To get started, have a look at the demo scripts "[ML_demo.m](https://github.com/JoramSoch/ML4ML/blob/main/ML_demo.m)" and "[ML4NI_demo.m](https://github.com/JoramSoch/ML4ML/blob/main/ML4NI_demo.m)".

Generally, an analysis proceeds in the following steps: Given features `Y`, classes `x` or targets `x`,

1. create a cross-validation (CV) matrix by calling `CV = ML_CV(x, k, mode)` where `k` is the number of cross-validation folds and `mode` is the desired cross-validation strategy (type `help ML_CV` for more info); **AND** <br>
2. a) perform support vector classification (SVC) by calling `SVC = ML_SVC(x, Y, CV, C)` **OR** <br>
   b) perform support vector regression (SVR) by calling `SVR = ML_SVR(x, Y, CV, C)` <br>
      where `C` is the SVM cost parameter and `CV` is the cross-validation matrix;
3. call `ML_SVM_res(SVC, 'perf')` or `ML_SVM_res(SVR, 'perf')` to visualize decoding results;
4. add input arguments `perm` and `subs` (SVC only) to generate permutations and/or perform subsampling.

To directly apply SVC or SVR to neuroimaging data (i.e. scans), use the function "[ML4NI_SVM.m](https://github.com/JoramSoch/ML4ML/blob/main/ML4NI_SVM.m)" (type `help ML4NI_SVM` for more info or [see below](https://github.com/JoramSoch/ML4ML#support-vector-machines-for-neuroimaging-data)).


## Documentation

### Cross-validation folds

```matlab
function CV = ML_CV(c, k, mode)
% _
% Cross-Validation Folds for Machine Learning Analysis
% FORMAT CV = ML_CV(c, k, mode)
% 
%     c    - an n x 1 vector of class labels (1, 2, 3 etc.)
%     k    - an integer larger than 1, the number of CV folds
%     mode - a string indicating the cross-validation mode
%            o 'kf'   - k-folds cross-validation across all points
%            o 'kfc'  - k-folds cross-validation on points per class
%            o 'loo'  - leave-one-out cross-validation across all points
%            o 'looc' - leave-one-out cross-validation on points per class
% 
%     CV   - an n x k matrix indicating training (1) and test (2)
%            data for each cross-validation fold
```

### Support vector classification

```matlab
function SVC = ML_SVC(x, Y, CV, C, perm, subs)
% _
% Cross-Validated Support Vector Machine for Classification
% FORMAT SVC = ML_SVC_ext(x, Y, CV, C, perm, subs)
% 
%     x    - an n x 1 vector of class labels (1, 2, 3 etc.)
%     Y    - an n x v matrix of predictor variables
%     CV   - an n x k matrix of cross-validation folds
%     C    - a scalar, the cost parameter of the SVM
%     perm - an integer, the number of permutations
%     subs - an integer, the number of subsamples
% 
%     SVC  - a structure specifying the calibrated SVM
%            o data - the data for the SVM (x, Y)
%                     o m     - the number of classes (=max(x))
%                     o N     - a 1 x m vector, number of points per class
%            o pars - parameters of the SVM (CV, C)
%                     o opt   - options for LibSVM's svmtrain
%                     o perm  - the number of permutations
%                     o subs  - the number of subsamples
%            o pred - predictions of the SVM
%                     o is    - an ne x subs matrix of subsampling indices
%                     o ip    - an ne x perm x subs array of permutation indices
%                     o xt    - an ne x perm x subs array of true class labels
%                     o xp    - an ne x perm x subs vector of predicted class labels
%                     ( where ne is the effective number of data points, i.e.
%                       n, if subs == 0; m x [min(N)-mod(min(N),10)], if subs > 1 )
%            o perf - predictive performance of the SVM
%                     o DA    - a  1 x perm x subs array of decoding accuracies
%                     o BA    - a  1 x perm x subs array of balanced accuracies
%                     o CA    - an m x perm x subs array of class accuracies
%                     o DA_CI - a  1 x 2 x subs array of 90% confidence intervals for DA
%                     o BA_CI - a  1 x 2 x subs array of 90% confidence intervals for BA
%                     o CA_CI - an m x 2 x subs array of 90% confidence intervals for CA
%                     o DA_pp - the permutation p-value for decoding accuracy
%                     o BA_pp - the permutation p-value for balanced accuracy
%                     o CA_pp - the permutation p-values for class accuracies
%                     o CM    - an m x m x subs array of confusion matrices
%                     ( CM(c2,c1,i) corresponds to the proportion of observations
%                       from class c1 classified into class c2 in subsample i )
```

### Support vector regression

```matlab
function SVR = ML_SVR(x, Y, CV, C, perm)
% _
% Cross-Validated Support Vector Machine for Regression
% FORMAT SVR = ML_SVR_ext(x, Y, CV, C, perm)
% 
%     x    - an n x 1 vector of target values
%     Y    - an n x v matrix of predictor variables
%     CV   - an n x k matrix of cross-validation folds
%     C    - a scalar, the cost parameter of the SVM
%     perm - an integer, the number of permutations
% 
%     SVR  - a structure specifying the calibrated SVM
%            o data - the data for the SVM (x, Y)
%            o pars - parameters of the SVM (CV, C)
%                     o opt  - options for LibSVM's svmtrain
%                     o perm - the number of permutations
%            o pred - predictions of the SVM
%                     o ip   - an n x perm matrix of permutation indices
%                     o xt   - an n x perm matrix of true target values
%                     o xp   - an n x perm matrix of predicted target values
%            o perf - predictive performance of the SVM
%                     o r    - a  1 x perm vector of predictive correlations
%                     o r_p  - parametric p-value for predictive correlation
%                     o r_CI - 90% confidence interval for predictive correlation
%                     o r_pp - permutation p-value for predictive correlation
%                     o r_cv - 90% critical values for predictive correlation
%                     o R2   - the coefficient of determination (=r^2, "R-squared")
%                     o MAE  - mean absolute error between true and predicted
%                     o MSE  - mean squared error between true and predicted
%                     o m    - slope of the line going through points (xt,xp)
%                     o n    - intercept of the line going through points (xt,xp)
```

### Support vector machines for neuroimaging data

```matlab
function ML4NI_SVM(SVM_dir, mask_img, data_imgs, c, x, X, preproc, options)
% _
% Support Vector Machine for Neuroimaging Data
% FORMAT ML4NI_SVM(SVM_dir, mask_img, data_imgs, c, x, X, preproc, options)
%     SVM_dir   - a string indicating the results directory
%     mask_img  - a string indicating the mask image (optional)
%     data_imgs - an N x 1 cell array of data image filepaths
%     c         - an N x 1 vector of class labels (1, 2, 3 etc.)
%     x         - an N x 1 vector of target values (real numbers)
%     X         - an N x p matrix of additional covariates (optional)
%     preproc   - a  1 x S structure of preprocessing steps
%                 o op  - a string indicating the operation (see below)
%                 o cov - a vector indexing covariates to use (from X)
%     options   - a structure specifying SVM options
%                 o SVM_type - a string indicating SVM type ('SVC' or 'SVR')
%                 o C        - a scalar, the SVM hyper-parameter
%                 o CV_mode  - a string indicating CV mode (see "help ML_CV")
%                 o k        - an integer, the number of CV folds
%                 o perm     - an integer, the number of permutations
%                 o subs     - an integer, the number of subsamples (for SVC)
```
