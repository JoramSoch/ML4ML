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
%                    o m     - the number of classes (=max(x))
%                    o N     - a 1 x m vector, number of points per class
%           o pars - parameters of the SVM (CV, C)
%                    o opt   - options for LibSVM's svmtrain
%           o pred - predictions of the SVM
%                    o xt    - an n x 1 vector of true class labels
%                    o xp    - an n x 1 vector of predicted class labels
%                    o xt_nz - vector of non-zero true class labels only
%                    o xp_nz - vector of predictions, where truth non-zero
%           o perf - predictive performance of the SVM
%                    o DA    - decoding accuracy
%                    o BA    - balanced accuracy
%                    o CA    - class accuracies, a 1 x m vector
%                    o DA_CI - 90% confidence interval for DA
%                    o BA_CI - 90% confidence interval for BA
%                    o CA_CI - 90% confidence interval for CA, a 2 x m matrix
%                    o CM    - confusion matrix, an m x m matrix
% 
% FORMAT SVC = ML_SVC(x, Y, CV, C) splits class labels x and predictor
% variables Y into cross-validation folds according to CV and calls LibSVM
% to perform support vector classification with cost parameter C. For this,
% LibSVM [1] should be on the MATLAB path.
% 
% References:
% [1] Chang C, Lin CJ. LIBSVM - A Library for Support Vector Machines;
%     URL: https://www.csie.ntu.edu.tw/~cjlin/libsvm/.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 06/07/2021, 14:27
%  Last edit: 17/08/2021, 18:14


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(CV), CV = ML_CV(x, 10, 'kfc'); end;
if nargin < 4 || isempty(C),  C  = 1; end;

% Get machine dimensions
%-------------------------------------------------------------------------%
n = size(CV,1);
k = size(CV,2);

% Get number of classes
%-------------------------------------------------------------------------%
m = max(x);
N = sum( repmat(x,[1 m])==repmat([1:m],[n 1]) );

% Prepare analysis display
%-------------------------------------------------------------------------%
fprintf('\n');
fprintf('-> Support vector classification:\n');
fprintf('   - %d x 1 class vector (%d classes);\n', n, m);
fprintf('   - %d x %d feature matrix;\n', n, size(Y,2));
fprintf('   - k = %d CV folds;\n', k);
fprintf('   - C = %g.\n', C);
fprintf('\n');
fprintf('-> Cross-validated predicition:\n');

% Cross-validated prediction
%-------------------------------------------------------------------------%
xt = zeros(n,1);                % true classes
xp = zeros(n,1);                % predicted classes
opt= sprintf('-s 0 -t 0 -c %s -q', num2str(C));
for g = 1:k                     % LibSVM options
    fprintf('   - CV fold %d ... ', g);
    % get test and training data
    i1 = find(CV(:,g)==1);
    i2 = find(CV(:,g)==2);
    x1 = x(i1);
    x2 = x(i2);
    Y1 = Y(i1,:);
    Y2 = Y(i2,:);
    % train and test using SVC
    svm1   = svmtrain(x1, Y1, opt);
    xp(i2) = svmpredict(x2, Y2, svm1, '-q');
    xt(i2) = x2;
    fprintf('successful!\n');
end;
clear i1 i2 x1 x2 Y1 Y2 svm1
fprintf('\n');

% Calculate performance
%-------------------------------------------------------------------------%
xp_nz = xp(xt~=0);              % remove missing data points
xt_nz = xt(xt~=0);
n_nz  = zeros(1,m);             % numbers of non-zeros
DA    = mean(xp_nz==xt_nz);     % decoding accuracy
CA    = zeros(1,m);             % class accuracies
CM    = zeros(m,m);             % confusion matrix
for j = 1:m
    n_nz(j) = sum(xt==j);       
    CA(j)   = mean(xp_nz(xt_nz==j)==xt_nz(xt_nz==j));
    CM(:,j) = mean(repmat(xp_nz(xt_nz==j),[1 m])==repmat([1:m],[sum(xt_nz==j) 1]))';
end;
BA = mean(CA);                  % balanced accuracy
[ph, ci1] = binofit(uint16(round(DA*sum(n_nz))), sum(n_nz), 0.1);
[ph, ci2] = binofit(uint16(floor(BA*sum(n_nz))), sum(n_nz), 0.1);
[ph, ci3] = binofit(uint16(round(CA.*n_nz)),     n_nz,      0.1);
DA_CI = ci1;                   % confidence intervals
BA_CI = ci2;
CA_CI = ci3;
clear ph ci1 ci2 ci3

% Assemble SVC structure
%-------------------------------------------------------------------------%
SVC.is_SVC     = true;          % support vector classification
SVC.data.x     = x;
SVC.data.Y     = Y;
SVC.data.m     = m;
SVC.data.N     = N;
SVC.pars.CV    = CV;
SVC.pars.C     = C;
SVC.pars.opt   = opt;
SVC.pred.xt    = xt;
SVC.pred.xp    = xp;
SVC.pred.xt_nz = xt_nz;
SVC.pred.xp_nz = xp_nz;
SVC.perf.DA    = DA;
SVC.perf.BA    = BA;
SVC.perf.CA    = CA;
SVC.perf.DA_CI = DA_CI;
SVC.perf.BA_CI = BA_CI;
SVC.perf.CA_CI = CA_CI;
SVC.perf.CM    = CM;