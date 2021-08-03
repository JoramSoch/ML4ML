# ML4ML

### Machine Learning Analysis for MATLAB

_A MATLAB Wrapper for Machine Learning, with another wrapper for neuroimaging data analysis_


## Requirements

These functions require the following packages:

* [LibSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) (for `ML_SVC` and `ML_SVR`) for MATLAB
* [SPM](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) (for `ML4NI_SVM` and `ML4NI_mask`), preferably [SPM12](https://github.com/spm/spm12).


## Getting Started

To get started, have a look at the demo scripts "[ML_demo.m](https://github.com/JoramSoch/ML4ML/blob/main/ML_demo.m)" and "[ML4NI_demo.m](https://github.com/JoramSoch/ML4ML/blob/main/ML4NI_demo.m)".

Generally, an analysis proceeds in two steps: Given features `Y`, classes `x` or targets `x`,

1. create a cross-validation (CV) matrix by calling `CV = ML_CV(x, k, mode)` where `k` is the number of cross-validation folds and `mode` is the desired cross-validation strategy (type `help ML_CV` for more info); **AND** <br>
2. a) perform support vector classification (SVC) by calling `SVC = ML_SVC(x, Y, CV, C)` **OR** <br>
   b) perform support vector regression (SVR) by calling `SVR = ML_SVR(x, Y, CV, C)` <br>
      where `C` is the SVM cost parameter and `CV` is the cross-validation matrix.

To directly apply SVC or SVR to neuroimaging data (i.e. scans), use the function "[ML4NI_SVM.m](https://github.com/JoramSoch/ML4ML/blob/main/ML4NI_SVM.m)" (type `help ML4NI_SVM` for more info or [see below](https://github.com/JoramSoch/ML4ML#support-vector-machines-for-neuroimaging-data)).


## Documentation

### Support vector classification

```matlab
function SVC = ML_SVC(x, Y, CV, C)
% _
% Cross-Validated Support Vector Machine for Classification
% FORMAT SVC = ML_SVC(x, Y, CV, C)
% 
%     x   - an n x 1 vector of class labels (1, 2, 3 etc.)
%     Y   - an n x v matrix of predictor variables
%     CV  - an n x k matrix of cross-validation folds
%     C   - a scalar, the cost parameter of the SVM
% 
%     SVC - a structure specifying the calibrated SVM
%           o data - the data for the SVM (x, Y)
%           o pars - parameters of the SVM (CV, C)
%           o pred - predictions of the SVM
%                    o xt    - an n x 1 vector of true class labels
%                    o xp    - an n x 1 vector of predicted class labels
%                    o xt_nz - vector of non-zero true class labels only
%                    o xp_nz - vector of predictions, where truth non-zero
%           o perf - predictive performance of the SVM
%                    o DA    - decoding accuracy
%                    o BA    - balanced accuracy
%                    o CA    - class accuracies, a 1 x max(x) vector
%                    o DA_CI - 90% confidence interval for DA
%                    o BA_CI - 90% confidence interval for BA
%                    o CA_CI - 90% confidence interval for CA, a 2 x max(x) matrix
```

### Support vector regression

```matlab
function SVR = ML_SVR(x, Y, CV, C)
% _
% Cross-Validated Support Vector Machine for Regression
% FORMAT SVR = ML_SVR(x, Y, CV, C)
% 
%     x   - an n x 1 vector of target values
%     Y   - an n x v matrix of predictor variables
%     CV  - an n x k matrix of cross-validation folds
%     C   - a scalar, the cost parameter of the SVM
% 
%     SVR - a structure specifying the calibrated SVM
%           o data - the data for the SVM (x, Y)
%           o pars - parameters of the SVM (CV, C)
%           o pred - predictions of the SVM
%                    o xt    - an n x 1 vector of true target values
%                    o xp    - an n x 1 vector of predicted target values
%                    o xt_nn - vector of non-NaN true target values only
%                    o xp_nn - vector of predictions, where truth non-NaN
%           o perf - predictive performance of the SVM
%                    o r     - correlation of true and predicted target values
%                    o r_CI  - 90% confidence interval for predictive correlation
%                    o R2    - the coefficient of determination (=r^2, "R-squared")
%                    o MAE   - mean absolute error between true and predicted
%                    o MSE   - mean squared error between true and predicted
%                    o m     - slope of the line going through points (xt,xp)
%                    o n     - intercept of the line going through points (xt,xp)
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
```
