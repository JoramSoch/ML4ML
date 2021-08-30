function SVC = ML_SVC_ext(x, Y, CV, C, perm, subs)
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
% 
% FORMAT SVC = ML_SVC(x, Y, CV, C) splits class labels x and predictor
% variables Y into cross-validation folds according to CV and calls LibSVM
% to perform support vector classification with cost parameter C. For this,
% LibSVM [1] should be on the MATLAB path.
% 
% If perm is larger than 1, target values x are permuted a number of times
% to establish a null distribution for the decoding accuracy.
% 
% If subs is larger than 0, class labels x are subsampled a number of times
% to avoid problems with unequal class sizes (i.e. the same number of points
% from each class is drawn repeatedly). Note that in this case, the variable
% "CV" is overwritten with "k-folds on points per class", accounting for
% the obtained subsamples (i.e. it is different in each subsample).
% 
% References:
% [1] Chang C, Lin CJ. LIBSVM - A Library for Support Vector Machines;
%     URL: https://www.csie.ntu.edu.tw/~cjlin/libsvm/.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 06/07/2021, 14:27
%  Last edit: 30/08/2021, 12:50


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(CV),   CV   = ML_CV(x, 10, 'kfc'); end;
if nargin < 4 || isempty(C),    C    = 1;     end;
if nargin < 5 || isempty(perm), perm = 1e5;   end;
if nargin < 6 || isempty(subs), subs = false; end;

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
fprintf('   - C = %g;\n', C);
fprintf('   - %d permutation(s);\n', perm);
fprintf('   - %d subsample(s).\n', subs);
fprintf('\n');
fprintf('-> Cross-validated predicition:\n');

% Prepare subsamples
%-------------------------------------------------------------------------%
rng(m);                         % initialize with number of classes
subs = max([0 subs]);
if subs == 0                    % no subsampling: use all points
    is = [1:n]';
else                            % subsampling: draw equal classes
    if mod(min(N),10) == 0
        spC = min(N) - 10;      % samples per class: round down to full 10's
    else
        spC = min(N) - mod(min(N),10);
    end;
    ic = cell(1,m);             % class indices: points in classes
    for j = 1:m
        ic{j} = find(x==j);
    end;
    is = zeros(m*spC,subs);     % subsample indices: points in subsamples
    for i = 1:subs
        isi = [];
        for j = 1:m             % for each class, take first spC indices
            icj = randperm(N(j))';          % after randomly permuting
            isi = [isi; ic{j}(icj(1:spC))]; % indices within this class
        end;
        is(:,i) = sort(isi);    % sort indices in ascending order
    end;
    clear isi icj
end;

% Prepare permutations
%-------------------------------------------------------------------------%
rng(n);                         % initialize with number of points
perm = max([1 perm]);
if subs == 0                    % no subsampling: use all points
    ip      = zeros(n,perm);
    ip(:,1) = [1:n]';           % neutral permutation
    for j = 2:perm
        ip(:,j) = randperm(n)'; % actual permutations
    end;
else                            % subsampling: use subsamples
    ip = zeros(m*spC,perm,subs);
    for i = 1:subs
        ip(:,1,i) = is(:,i);    % neutral permutation
        for j = 2:perm          % actual permutations
            ip(:,j,i) = is(randperm(m*spC)',i);
        end;
    end;
end;

% Prepare cross-validation
%-------------------------------------------------------------------------%
if subs == 0
    subs    = 1;
    subsamp = false;
else
    subsamp = true;
    CV_orig = CV;
    CV      = zeros(m*spC,k,subs);
end;

% Cross-validated prediction
%-------------------------------------------------------------------------%
xt = zeros(size(ip));           % "true" classes
xp = zeros(size(ip));           % predicted classes
opt= sprintf('-s 0 -t 0 -c %s -q', num2str(C));
for i = 1:subs                  % LibSVM options
    % obtain new CV matrix
    if subsamp
        fprintf('   - subsample %d:\n', i);
        CV(:,:,i) = ML_CV(x(is(:,i)), k, 'kfc');
        % alternatively, use subsample-compliant
        % subset of originally given CV matrix:
        % CV(:,:,i) = CV_orig(is(:,i),:);
    end;
    % perform cross-validation
    for g = 1:k
        if subsamp, fprintf('  '); end;
        fprintf('   - CV fold %d: permutation ', g);
        % get test and training indices
        i1 = find(CV(:,g,i)==1);
        i2 = find(CV(:,g,i)==2);
        % get test and training targets
        x1 = x(is(i1,i));
        x2 = x(is(i2,i));
        % analyze permuted data
        for j = 1:perm
            if j == 1 || mod(j,floor(perm/10)) == 0 || j == perm
                fprintf('%d, ', j);
            end;
            % get test and training data
            Y1 = Y(ip(i1,j,i),:);
            Y2 = Y(ip(i2,j,i),:);
            % train and test using SVC
            svm1       = svmtrain(x1, Y1, opt);
            xp(i2,j,i) = svmpredict(x2, Y2, svm1, '-q');
            xt(i2,j,i) = x2;
        end;
        fprintf('done.\n');
    end;
end;
clear i1 i2 x1 x2 Y1 Y2 svm1
fprintf('\n');

% Calculate performance
%-------------------------------------------------------------------------%
ne = size(xp,1);                % effective number of data points
if ~subsamp, nc = N';           % number of data points per class
else,        nc = spC*ones(m,1); end;
DA    = zeros(1,perm,subs);
BA    = zeros(1,perm,subs);
CA    = zeros(m,perm,subs);
DA_CI = zeros(1,2,subs);
BA_CI = zeros(1,2,subs);
CA_CI = zeros(m,2,subs);
CM    = zeros(m,m,subs);
for i = 1:subs
    for j = 1:perm              % decoding accuracy
        DA(1,j,i) = mean(xp(:,j,i)==xt(:,j,i));
        for h = 1:m             % class accuracies
            CA(h,j,i) = mean(xp(xt(:,j,i)==h,j,i)==h);
        end;                    % balanced accuracy
        BA(1,j,i) = mean(CA(:,j,i));
    end;                        % 90% confidence intervals
    [ph, DA_CI(1,:,i)] = binofit(uint16(round(DA(1,1,i)*ne)),  ne, 0.1);
    [ph, BA_CI(1,:,i)] = binofit(uint16(floor(BA(1,1,i)*ne)),  ne, 0.1);
    [ph, CA_CI(:,:,i)] = binofit(uint16(round(CA(:,1,i).*nc)), nc, 0.1);
    for h = 1:m                 % confusion matrix
        CM(:,h,i) = mean( repmat(xp(xt(:,1,i)==h,1,i),[1 m])==repmat([1:m],[sum(xt(:,1,i)==h) 1]) )';
    end;
end;                            % permutation p-values
DA_pp = sum(mean(DA,3)>=mean(DA(1,1,:),3))/perm;
BA_pp = sum(mean(BA,3)>=mean(BA(1,1,:),3))/perm;
CA_pp = sum(mean(CA,3)>=repmat(mean(CA(:,1,:),3),[1 perm]),2)/perm;
clear ne nc ph

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
SVC.pars.perm  = perm;
SVC.pars.subs  = (~subsamp)*0 + (subsamp)*subs;
SVC.pred.is    = is;
SVC.pred.ip    = ip;
SVC.pred.xt    = xt;
SVC.pred.xp    = xp;
SVC.perf.DA    = DA;
SVC.perf.BA    = BA;
SVC.perf.CA    = CA;
SVC.perf.DA_CI = DA_CI;
SVC.perf.BA_CI = BA_CI;
SVC.perf.CA_CI = CA_CI;
SVC.perf.DA_pp = DA_pp;
SVC.perf.BA_pp = BA_pp;
SVC.perf.CA_pp = CA_pp;
SVC.perf.CM    = CM;