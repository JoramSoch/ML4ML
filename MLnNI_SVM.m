function MLnNI_SVM(SVM_dir, c, x, X, preproc, options)
% _
% Support Vector Machine for Non-Neuroimaging Data
% FORMAT ML4NI_SVM(SVM_dir, c, x, X, preproc, options)
%     SVM_dir   - a string indicating the results directory
%     c         - an N x 1 vector of class labels (1, 2, 3 etc.)
%     x         - an N x 1 vector of target values (real numbers)
%     X         - an N x p matrix of predictor variables
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
% 
% FORMAT ML4NI_SVM(SVM_dir, c, x, X, preproc, options) performs an SVM
% analysis specified via c, x and X with preprocessing operations "preproc"
% and SVM specifications "options" and saves SVM results into SVM_dir.
% 
% Preprocessing steps are carried in out in the order in which they are
% specified. For details, see "ML_preproc.m" or type "help ML_preproc".
% 
% The default configuration for preprocessing steps is that all covariates
% are mean-centered and then added to the feature matrix, i.e.
%     preproc(1).op  = 'mc_X'
%     preproc(1).cov = []
%     preproc(2).op  = 'add_X'
%     preproc(2).cov = [1:p]
% Note: If the supplied structure "preproc" is non-empty, these operations
% are not automatically performed and must be specified!
% 
% The default values for the fields in SVM options are:
% - SVM_type: 'SVC', if the input variable "x" is empty; 'SVR' otherwise
% - C       : C = 1
% - CV_mode : 'kfc', if the input variable "c" is non-empty; 'kf' otherwise
% - k       : k = 10
% - perm    : perm = 1
% - subs    : subs = 0
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 27/07/2021, 07:21
%  Last edit: 13/10/2021, 16:00


% Set default values (c, x)
%-------------------------------------------------------------------------%
if isempty(c) || nargin < 2, c = []; end;   % no class labels
if isempty(x) || nargin < 3, x = []; end;   % no target values

% Set default values (preproc)
%-------------------------------------------------------------------------%
if isempty(preproc) || nargin < 5
    preproc(1).op  = 'mc_X';                % mean-center features
    preproc(1).cov = [1:size(X,2)];
    preproc(2).op  = 'add_X';               % add covariates to features
    preproc(2).cov = [1:size(X,2)];
end;

% Set default values (options)
%-------------------------------------------------------------------------%
if isempty(options) || nargin < 6           % create options structure
    options = struct();
end;
if ~isempty(options)
    if ~isfield(options,'SVM_type')         % support vector machine type
        if isempty(x), options.SVM_type = 'SVC';
        else,          options.SVM_type = 'SVR'; end;
    end;
    if ~isfield(options,'CV_mode')          % cross-validation mode
        if ~isempty(c), options.CV_mode = 'kfc';
        else,           options.CV_mode = 'kf';  end;
    end;
    if ~isfield(options,'C'),    options.C    = 1; end; % SVM hyper-parameter
    if ~isfield(options,'k'),    options.k    = 10;end; % number of CV folds
    if ~isfield(options,'perm'), options.perm = 1; end; % number of permutations
    if ~isfield(options,'subs'), options.subs = 0; end; % number of subsamples
end;

% Preprocess features
%-------------------------------------------------------------------------%
 Y     = [];                    % empty feature matrix
[Y, x] = ML_preproc(Y, c, x, X, preproc);

% Perform SVM analysis
%-------------------------------------------------------------------------%
if ~isempty(c)                  % CV on points per class
    CV = ML_CV(c, options.k, options.CV_mode);
else                            % CV across all points
    CV = ML_CV(size(Y,1), options.k, options.CV_mode);
end;
if strcmp(options.SVM_type,'SVC')           % support vector classification
    SVM = ML_SVC(c, Y, CV, options.C, options.perm, options.subs);
end;
if strcmp(options.SVM_type,'SVR')           % support vector regression
    SVM = ML_SVR(x, Y, CV, options.C, options.perm);
end;

% Save SVM results
%-------------------------------------------------------------------------%
if ~exist(SVM_dir,'dir'), mkdir(SVM_dir); end;
save(strcat(SVM_dir,'/','SVM.mat'), 'SVM');

% Save SVM design
%-------------------------------------------------------------------------%
design.SVM_dir   = SVM_dir;
design.c         = c;
design.x         = x;
design.X         = X;
design.preproc   = preproc;
design.options   = options;
save(strcat(SVM_dir,'/','design.mat'), 'design');