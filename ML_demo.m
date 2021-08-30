% Machine Learning Demo
% _
% Demo Script for the Machine Learning Toolkit
% FORMAT ML_demo;
% 
% Attention: The entire script works with random data.
% Please amend/adapt for your own requirements.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 20/07/2021, 16:45
%  Last edit: 30/08/2021, 15:50


clear

%%% Step 0: some general parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify dimensions
n = 200;            % number of data points
q = 3;              % number of classes
v = 100;            % number of features

% specify options
k    = 10;          % number of cross-validation folds
C    = 1;           % hyper-parameter for SVM calibration
perm = 100;         % number of permutations
subs = 0;           % number of subsamples


%%% Example 1: support vector classification (SVC) %%%%%%%%%%%%%%%%%%%%%%%%

% x - an n x 1 vector of class labels (1, 2, 3 etc.)
x = randi([1,q],[n,1]);

% Y - an n x v matrix of predictor variables ("features")
Y = randn(n,v);

% CV - an n x k matrix of cross-validation folds
CV = ML_CV(x, k, 'kfc');

% SVC - a structure specifying the calibrated SVM
SVC = ML_SVC(x, Y, CV, C, perm, subs);

% SVC results: decoding accuracy etc.
ML_SVM_res(SVC, 'perf');
% for other results, type "help ML_SVM_res"


%%% Example 2: support vector regression (SVR) %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% x - an n x 1 vector of target values (real numbers)
x = rand(n,1);
c = ones(n,1);      % assuming just one class

% Y - an n x v matrix of predictor variables ("features")
Y = randn(n,v);

% CV - an n x k matrix of cross-validation folds
CV = ML_CV(c, k, 'kf');

% SVR - a structure specifying the calibrated SVM
SVR = ML_SVR(x, Y, CV, C, perm);

% SVR results: predictive correlation etc.
ML_SVM_res(SVR, 'perf');
% for other results, type "help ML_SVM_res"


%%% Example 3: distributional transformation (DT) %%%%%%%%%%%%%%%%%%%%%%%%%

% DT - a post-processing operation preserving the distribution of the target variable
% xd = predicted target values, distributionally transformed to the training sample
SVR.pred.xd  = ML_DT(SVR.pred.xt(:,1), SVR.pred.xp(:,1), SVR.pars.CV);

% r_DT - predictive correlation using distributional transformation
[a, b, c, d] = corrcoef(SVR.pred.xd, SVR.pred.xt(:,1), 'Alpha', 0.1);
r_DT    =  a(1,2);
r_CI_DT = [c(1,2), d(1,2)];
clear a b c d