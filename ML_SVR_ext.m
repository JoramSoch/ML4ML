function SVR = ML_SVR_ext(x, Y, CV, C, perm)
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
% 
% FORMAT SVR = ML_SVC(x, Y, CV, C) splits target values x and predictor
% variables Y into cross-validation folds according to CV and calls LibSVM
% to perform support vector regression with cost parameter C. For this,
% LibSVM [1] should be on the MATLAB path.
% 
% If perm is larger than 1, target values x are permuted a number of times
% to establish a null distribution for the predictive correlation.
% 
% References:
% [1] Chang C, Lin CJ. LIBSVM - A Library for Support Vector Machines;
%     URL: https://www.csie.ntu.edu.tw/~cjlin/libsvm/.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 06/07/2021, 14:51
%  Last edit: 25/08/2021, 12:33


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(CV),   CV   = ML_CV(numel(x), 10, 'kf'); end;
if nargin < 4 || isempty(C),    C    = 1;   end;
if nargin < 5 || isempty(perm), perm = 1e5; end;

% Get machine dimensions
%-------------------------------------------------------------------------%
n = size(CV,1);
k = size(CV,2);

% Prepare analysis display
%-------------------------------------------------------------------------%
fprintf('\n');
fprintf('-> Support vector regression:\n');
fprintf('   - %d x 1 target vector;\n', n);
fprintf('   - %d x %d feature matrix;\n', n, size(Y,2));
fprintf('   - k = %d CV folds;\n', k);
fprintf('   - C = %g;\n', C);
fprintf('   - %d permutation(s).\n', perm);
fprintf('\n');
fprintf('-> Cross-validated predicition:\n');

% Prepare permutations
%-------------------------------------------------------------------------%
rng(n);                         % initialize with number of points
perm    = max([1 perm]);
ip      = zeros(n,perm);
ip(:,1) = [1:n]';               % neutral permutation
for j = 2:perm
    ip(:,j) = randperm(n)';     % actual permutations
end;

% Cross-validated prediction
%-------------------------------------------------------------------------%
xt = zeros(n,perm);             % "true" targets
xp = zeros(n,perm);             % predicted targets
opt= sprintf('-s 4 -t 0 -c %s -q', num2str(C));
for j = 1:perm                  % LibSVM options
    fprintf('   - permutation %d: CV fold ', j);
    % permute cross-validation matrix
    CV_j = CV(ip(:,j),:);
    % perform cross-validation
    for g = 1:k
        fprintf('%d, ', g);
        % get test and training indices
        i1 = find(CV_j(:,g)==1);
        i2 = find(CV_j(:,g)==2);
        % get test and training targets
        x1 = x(ip(i1,j));
        x2 = x(ip(i2,j));
        % get test and training data
        Y1 = Y(i1,:);
        Y2 = Y(i2,:);
        % train and test using SVC
        svm1     = svmtrain(x1, Y1, opt);
        xp(i2,j) = svmpredict(x2, Y2, svm1, '-q');
        xt(i2,j) = x2;
    end;
    fprintf('done.\n');
end;
clear CV_j i1 i2 x1 x2 Y1 Y2 svm1
fprintf('\n');

% Calculate performance
%-------------------------------------------------------------------------%
[a, b, c, d] = corrcoef(xp(:,1), xt(:,1), 'Alpha', 0.1);
r    = [a(1,2), zeros(1,perm-1)];           % correlation coefficient
r_p  =  b(1,2);                             % correlation p-value
r_CI = [c(1,2), d(1,2)];                    % 90% confidence interval
clear a b c d
for j = 2:perm
    r(j) = corr(xp(:,j), xt(:,j));          % permuted correlations
end;
rs   = sort(r);                             % sorted correlations
pp   = sum(r>=r(1))/perm;                   % permutation p-value
if perm > 2
    cv =[rs(ceil(1+0.05*perm)), ...         % 90% critical values
         rs(floor(perm-0.05*perm))];
else
    cv = [NaN, NaN];
end;
clear rs
R2   = r(1)^2;                              % coefficient of determination
MAE  = mean(abs(xp(:,1)-xt(:,1)));          % mean absolute error
MSE  = mean((xp(:,1)-xt(:,1)).^2);          % mean squared error
mn   = polyfit(xt(:,1),xp(:,1),1);          % slope and intercept

% Assemble SVR structure
%-------------------------------------------------------------------------%
SVR.is_SVC    = false;          % support vector regression
SVR.data.x    = x;
SVR.data.Y    = Y;
SVR.pars.CV   = CV;
SVR.pars.C    = C;
SVR.pars.opt  = opt;
SVR.pars.perm = perm;
SVR.pred.ip   = ip;
SVR.pred.xt   = xt;
SVR.pred.xp   = xp;
SVR.perf.r    = r;
SVR.perf.r_p  = r_p;
SVR.perf.r_CI = r_CI;
SVR.perf.r_pp = pp;
SVR.perf.r_cv = cv;
SVR.perf.R2   = R2;
SVR.perf.MAE  = MAE;
SVR.perf.MSE  = MSE;
SVR.perf.m    = mn(1);
SVR.perf.n    = mn(2);