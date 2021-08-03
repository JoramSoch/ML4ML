% Neuroimaging Analysis Demo
% _
% Demo Script for Machine Learning of Neuroimaging Data
% FORMAT ML4NI_demo;
% 
% Attention: The entire script works with random data and dummy variables.
% Please amend/adapt for your own requirements.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 03/08/2021, 12:03
%  Last edit: 03/08/2021, 12:03


clear

%%% Step 0: some general parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify dimensions
N = 200;            % number of data points
q = 3;              % number of classes

% specify options
k = 10;             % number of cross-validation folds
C = 1;              % hyper-parameter for SVM calibration


%%% Step 1: specify analysis parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SVM_dir - a string indicating the results directory
SVM_dir   = 'C:\analysis\';

% mask_img - a string indicating the mask image (optional)
mask_img  = [];
% Hint: leave empty for automatic mask generation.

% data_imgs - an n x 1 cell array of data image filepaths
data_imgs = cell(N,1);
for i = 1:N
    data_imgs{i} = sprintf('C:\data\scan_%03d.nii', i);
end;

% c - an n x 1 vector of class labels (1, 2, 3 etc.)
c = randi([1,q],[N,1]);

% x - an n x 1 vector of target values (real numbers)
x = rand(N,1);

% X - an n x p matrix of some other covariates
X = [randn(N,5), randi([0,1],[N,2])];


%%% Example 1: support vector classification (SVC) %%%%%%%%%%%%%%%%%%%%%%%%

% specify preprocessing
preproc(1).op  = 'mc_Y';        % mean-center feature matrix
preproc(1).cov = [];
preproc(2).op  = 'add_X';       % add covariates to feature matrix
preproc(2).cov = [1:5];

% specify SVM analysis
options.SVM_type = 'SVC';
options.C        = 1;
options.CV_mode  = 'kfc';
options.k        = 10;

% perform SVM analysis
ML4NI_SVM(SVM_dir, mask_img, data_imgs, c, [], X, preproc, options)


%%% Example 2: support vector regression (SVR) %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify preprocessing
preproc(1).op  = 'reg_Y';       % regress covs 6-7 out of feature matrix
preproc(1).cov = [6:7];
preproc(2).op  = 'mcc_Y';       % mean-center covs 1-5 by class
preproc(2).cov = [1:5];
preproc(3).op  = 'add_X';       % add covs 1-5 to feature matrix
preproc(3).cov = [1:5];
preproc(4).op  = 'mc_Y';        % mean-center feature matrix
preproc(4).cov = [];
preproc(5).op  = 'std_Y';       % standardize feature matrix
preproc(5).cov = [];

% specify SVM analysis
options.SVM_type = 'SVR';
options.C        = 1;
options.CV_mode  = 'kf';
options.k        = 10;

% perform SVM analysis
ML4NI_SVM(SVM_dir, mask_img, data_imgs, c, x, X, preproc, options)