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
% 
% FORMAT SVR = ML_SVC(x, Y, CV, C) splits target values x and predictor
% variables Y into cross-validation folds according to CV and calls LibSVM
% to perform support vector regression with cost parameter C. For this,
% LibSVM [1] should be on the MATLAB path.
% 
% References:
% [1] Chang C, Lin CJ. LIBSVM - A Library for Support Vector Machines;
%     URL: https://www.csie.ntu.edu.tw/~cjlin/libsvm/.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 06/07/2021, 14:51
%  Last edit: 03/08/2021, 11:13


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(CV)
    CV = ML_CV(numel(x), 10, 'kfc');
end;

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
fprintf('   - C = %g.\n', C);
fprintf('\n');
fprintf('-> Cross-validated predicition:\n');

% Cross-validated prediction
%-------------------------------------------------------------------------%
xt = zeros(n,1);                % true targets
xp = zeros(n,1);                % predicted targets
opt= sprintf('-s 4 -t 0 -c %s -q', num2str(C));
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
clear i1 i2 svm1
fprintf('\n');

% Calculate performance
%-------------------------------------------------------------------------%
xp_nn = xp(~isnan(xt),:);       % remove missing data points
xt_nn = xt(~isnan(xt));
[a, b, c, d] = corrcoef(xp_nn, xt_nn, 'Alpha', 0.1);
r     =  a(1,2);                % predictive correlation
r_CI  = [c(1,2); d(1,2)];       % 90% confidence interval
R2    = r^2;                    % coefficient of determination
MAE   = mean(abs(xp_nn-xt_nn)); % mean absolute error
MSE   = mean((xp_nn-xt_nn).^2); % mean squared error
mn    = polyfit(xt_nn,xp_nn,1); % slope and intercept
clear a b c d

% Assemble SVR structure
%-------------------------------------------------------------------------%
SVR.is_SVC     = false;         % support vector regression
SVR.data.x     = x;
SVR.data.Y     = Y;
SVR.pars.CV    = CV;
SVR.pars.C     = C;
SVR.pars.opt   = opt;           % options for LibSVM's svmtrain
SVR.pred.xt    = xt;
SVR.pred.xp    = xp;
SVR.pred.xt_nn = xt_nn;
SVR.pred.xp_nn = xp_nn;
SVR.perf.r     = r;
SVR.perf.r_CI  = r_CI;
SVR.perf.R2    = R2;
SVR.perf.MAE   = MAE;
SVR.perf.MSE   = MSE;
SVR.perf.m     = mn(1);
SVR.perf.n     = mn(2);