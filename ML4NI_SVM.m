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
% 
% FORMAT ML4NI_SVM(SVM_dir, mask_img, data_imgs, c, x, X, preproc, options)
% performs an SVM analysis specified via data_imgs, c, x and X with
% preprocessing operations "preproc" and SVM specifications "options" and
% saves SVM results as well as mask image into SVM_dir.
% 
% Note: If no mask image is specified, then a second-level mask is
% automatically generated as the intersection of all data images.
% 
% Preprocessing steps are carried in out in the order in which they are
% specified. Each entry of the vector "preproc" codes one operation, e.g.
%     preproc(1).op  = 'mc_X'
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
% - 'reg_Y' : regress selected covariates out of features matrix Y;
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
%  Last edit: 30/08/2021, 16:27


% Set default values (c, x, X)
%-------------------------------------------------------------------------%
if isempty(c) || nargin < 4, c = []; end;   % no class labels
if isempty(x) || nargin < 5, x = []; end;   % no target values
if isempty(X) || nargin < 6, X = []; end;   % no covariates

% Set default values (preproc)
%-------------------------------------------------------------------------%
if isempty(preproc) || nargin < 7
    preproc(1).op  = 'mc_Y';                % mean-center features
    preproc(1).cov = [];
    preproc(2).op  = 'add_X';               % add covariates to features
    preproc(2).cov = [1:size(X,2)];
end;

% Set default values (options)
%-------------------------------------------------------------------------%
if isempty(options) || nargin < 8           % create options structure
    options = struct();
end;
if ~isempty(options)
    if ~isfield(options,'SVM_type')         % support vector machine type
        if isempty(x), options.SVM_type = 'SVC';
        else,          options.SVM_type = 'SVR'; end;
    end;
    if ~isfield(options,'CV_mode')          % cross-validation mode
        if ~isempty(c), options.CV_mode = 'kfc';
        else,           options.CV_mode = 'kf'; end;
    end;
    if ~isfield(options,'C'),    options.C    = 1; end; % SVM hyper-parameter
    if ~isfield(options,'k'),    options.k    = 10;end; % number of CV folds
    if ~isfield(options,'perm'), options.perm = 1; end; % number of permutations
    if ~isfield(options,'subs'), options.subs = 0; end; % number of subsamples
end;

% Get data dimensions
%-------------------------------------------------------------------------%
N = numel(data_imgs);           % number of data images
H = spm_vol(data_imgs{1});      % header of data image
V = prod(H.dim);                % number of voxels

% Load data images
%-------------------------------------------------------------------------%
Finter = spm('FigName','ML4NI_SVM: load');
spm_progress_bar('Init',100,'Load data images...','');
Y = zeros(N,V);                 % feature matrix
for i = 1:N
    y_hdr = spm_vol(data_imgs{i});
    y_img = spm_read_vols(y_hdr);
    Y(i,:)= reshape(y_img,[1 V]);
    spm_progress_bar('Set',(i/N)*100);
end;
spm_progress_bar('Clear');
clear y_hdr y_img

% Load mask image
%-------------------------------------------------------------------------%
if ~isempty(mask_img)
    copyfile(mask_img, strcat(SVM_dir,'/','mask.nii'), 'f');
    m_hdr = spm_vol(mask_img);
    m_img = spm_read_vols(m_hdr);
    m_ind = find(m_img~=0);     % in-mask voxel indices
    clear m_hdr m_img
end;

% Create mask image
%-------------------------------------------------------------------------%
if isempty(mask_img)
    [m_img, m_hdr, m_ind] = ML4NI_mask(Y, H);
    mask_img      = strcat(SVM_dir,'/','mask.nii');
    m_hdr.fname   = mask_img;
    m_hdr.descrip = 'ML4NI_SVM: mask image';
    spm_write_vol(m_hdr,reshape(m_img,H.dim));
    clear m_hdr m_img
end;

% Preprocess features
%-------------------------------------------------------------------------%
Y = Y(:,m_ind);                 % restrict to in-mask voxels
for j = 1:numel(preproc)
    k = preproc(j).cov;         % covariate indices of current step
    if strcmp(preproc(j).op,'mc_Y')
        Y = Y - repmat(mean(Y),[N 1]);
    elseif strcmp(preproc(j).op,'std_Y')
        Y = Y./ repmat(std(Y),[N 1]);
    elseif strcmp(preproc(j).op,'mc_X')
        X(:,k) = X(:,k) - repmat(mean(X(:,k)),[N 1]);
    elseif strcmp(preproc(j).op,'std_X')
        X(:,k) = X(:,k)./ repmat(std(X(:,k)),[N 1]);
    elseif strcmp(preproc(j).op,'mcc_X')
        for h = 1:max(c)
            X(c==h,k) = X(c==h,k) - repmat( mean(X(c==h,k)), [sum(c==h), 1] );
        end;
    elseif strcmp(preproc(j).op,'stdc_X')
        for h = 1:max(c)
            X(c==h,k) = X(c==h,k)./ repmat( std(X(c==h,k)), [sum(c==h), 1] );
        end;
    elseif strcmp(preproc(j).op,'reg_x') || strcmp(preproc(j).op,'reg_Y')
        Xk = X(:,k);            % compute residual-forming matrix
        Rk = eye(N) - Xk*(Xk'*Xk)^(-1)*Xk';
        if strcmp(preproc(j).op,'reg_x')
            x = Rk*x;
        elseif strcmp(preproc(j).op,'reg_Y')
            Y = Rk*Y;
        end;
    elseif strcmp(preproc(j).op,'add_X')
        Y = [Y, X(:,k)];
    end;
end;

% Perform SVM analysis
%-------------------------------------------------------------------------%
if ~isempty(c)                  % CV on points per class
    CV = ML_CV(c, options.k, options.CV_mode);
else                            % CV across all points
    CV = ML_CV(ones(N,1), options.k, options.CV_mode);
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
design.mask_img  = mask_img;
design.data_imgs = data_imgs;
design.c         = c;
design.x         = x;
design.X         = X;
design.preproc   = preproc;
design.options   = options;
save(strcat(SVM_dir,'/','design.mat'), 'design');