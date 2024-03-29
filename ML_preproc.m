function [Y, x] = ML_preproc(Y, c, x, X, preproc)
% _
% Pre-Processing for Machine Learning Analysis
% FORMAT [Y, x] = ML_preproc(Y, c, x, X, preproc)
% 
%     Y - an N x v matrix of feature values (optional)
%     c - an N x 1 vector of class labels (1, 2, 3 etc.)
%     x - an N x 1 vector of target values (real numbers)
%     X - an N x p matrix of additional covariates (optional)
% 
%     Y - an N x w matrix of preprocessed features (see below)
%     x - an N x 1 vector of preprocessed targets (if applicable)
% 
% FORMAT [Y, x] = ML_preproc(Y, c, x, X, preproc) takes a feature matrix Y,
% class vector c, targets vector x and covariate matrix X and preprocesses
% the features Y and X according to steps defined in the struct "preproc".
% 
% For example, Y can be a matrix of imaging features (subject x voxel)
% and X can be a matrix of non-imaging covariates (subject x predictor)
% that is supposed to be added to the feature matrix. Usually, c is a
% vector of class labels used for support vector classification (SVC) and
% x is a vector of target values used for support vector regression (SVR).
% 
% Essentially, all of the input variables are optional. However, either
% Y or X must be provided for steps that depend on sample size, e.g.
% mean-centering and standardization. Note further that x is only outputed
% when x is also inputed and regression out of targets is selected as a
% preprocessed step.
% 
% Preprocessing steps are carried out in the order in which they are
% specified. Each entry of the vector "preproc" codes one operation, e.g.
%     preproc(1).op  = 'mcc_X'
%     preproc(1).cov = [1 2 3]
%     preproc(2).op  = 'reg_Y'
%     preproc(2).cov = [4 5 6]
% would mean-center covariates 1-3, separately by class, and regress
% covariates 4-6 out of the features, taking residuals for prediction.
% 
% The following operations are available as preprocessing steps:
% - 'mc_Y'  : mean-center all features by subtracting the mean;
% - 'std_Y' : standardize all features by dividing with the SD;
% - 'mc_X'  : mean-center selected covariates by subtracting the mean;
% - 'std_X' : standardize selected covariates by diving with the SD;
% - 'mcc_X' : mean-center selected covariates, separately by class;
% - 'stdc_X': standardize selected covariates, separately by class;
% - 'reg_x' : regress selected covariates out of target values x;
% - 'reg_Y' : regress selected covariates out of feature matrix Y;
% - 'add_X' : add selected covariates to the feature matrix Y.
% Note: The first two operations will ignore the field "cov" of "preproc".
% 
% The default configuration for preprocessing steps is that all features
% are mean-centered and all covariates are added to the feature matrix, i.e.
%     preproc(1).op  = 'mc_Y'
%     preproc(1).cov = []
%     preproc(2).op  = 'add_X'
%     preproc(2).cov = [1:p]
% Note: If the supplied structure "preproc" is non-empty, these operations
% are not automatically performed and must be specified!
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 27/07/2021, 07:21
%  Last edit: 18/10/2021, 15:52


% Set default values
%-------------------------------------------------------------------------%
if isempty(preproc) || nargin < 5
    preproc(1).op  = 'mc_Y';    % mean-center features
    preproc(1).cov = [];
    preproc(2).op  = 'add_X';   % add covariates to features
    preproc(2).cov = [1:size(X,2)];
end;

% Get sample size
%-------------------------------------------------------------------------%
if ~isempty(Y)
    N = size(Y,1);
elseif ~isempty(X)
    N = size(X,1);
elseif ~isempty(c)
    N = size(c,1);
elseif ~isempty(x)
    N = size(x,1);
end;

% Preprocess features
%-------------------------------------------------------------------------%
for j = 1:numel(preproc)
    k = preproc(j).cov;         % covariate indices of current step
    switch preproc(j).op
        case 'mc_Y'
            Y = Y - repmat(mean(Y,'omitnan'),[N 1]);
        case 'std_Y'
            Y = Y./ repmat(std(Y,'omitnan'),[N 1]);
        case 'mc_X'
            X(:,k) = X(:,k) - repmat(mean(X(:,k),'omitnan'),[N 1]);
        case 'std_X'
            X(:,k) = X(:,k)./ repmat(std(X(:,k),'omitnan'),[N 1]);
        case 'mcc_X'
            for h = 1:max(c)
                X(c==h,k) = X(c==h,k) - repmat( mean(X(c==h,k),'omitnan'), [sum(c==h), 1] );
            end;
        case 'stdc_X'
            for h = 1:max(c)
                X(c==h,k) = X(c==h,k)./ repmat( std(X(c==h,k),'omitnan'), [sum(c==h), 1] );
            end;
        case 'reg_x'
            Xk = X(:,k);        % compute residual-forming matrix
            Rk = eye(N) - Xk*(Xk'*Xk)^(-1)*Xk';
            x  = Rk*x;
        case 'reg_Y'
            Xk = X(:,k);        % compute residual-forming matrix
            Rk = eye(N) - Xk*(Xk'*Xk)^(-1)*Xk';
            Y  = Rk*Y;
        case 'add_X'
            Y = [Y, X(:,k)];
    end;
end;