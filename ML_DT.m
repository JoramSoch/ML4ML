function xd = ML_DT(xt, xp, CV)
% _
% Distributional Transformation after Machine Learning Analysis
% FORMAT xd = ML_DT(xt, xp, CV)
% 
%     xt - an n x 1 vector of true target values
%     xp - an n x 1 vector of predicted target values
%     CV - an n x k matrix of cross-validation folds
% 
%     xd - an n x 1 vector of distributionally transformed
%                             predicted target values [1,2]
% 
% FORMAT xd = ML_DT(xt, xp, CV), in every cross-validation fold CV,
% distributionally transforms test predictions xp to training samples xt.
% 
% Distributional transform is a post-processing method for prediction
% analyses which matches predicted values in the test data to observed
% values in the training data by matching their empirical cumulative
% distribution functions [2] which can increase decoding accuracy [1].
% 
% References:
% [1] Soch, Joram (2020): "Distributional Transformation Improves Decoding
%     Accuracy When Predicting Chronological Age From Structural MRI";
%     in: Frontiers in Psychiatry, vol. 11, art. 604268; URL: https://
%     www.frontiersin.org/articles/10.3389/fpsyt.2020.604268/full;
%     DOI: 10.3389/fpsyt.2020.604268.
% [2] JoramSoch (2021): "Distributional transformation using cumulative
%     distribution function"; in: The Book of Statistical Proofs; URL:
%     https://statproofbook.github.io/P/cdf-dt.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 06/07/2021, 15:13
%  Last edit: 02/08/2021, 15:08


% Get data dimensions
%-------------------------------------------------------------------------%
n = size(CV,1);
k = size(CV,2);

% Transform predictions
%-------------------------------------------------------------------------%
xd = zeros(size(xp));
for g = 1:k
    % get test and training data
    i1  = find(CV(:,g)==1);
    i2  = find(CV(:,g)==2);
    xt1 = xt(i1);               % true target values in training data
    xp2 = xp(i2);               % predicted target values in test data
    % distributional transformation
    [f1, x1] = ecdf(xp2);
    [f2, x2] = ecdf(xt1);
    xd2 = zeros(size(xp2));
    for i = 1:numel(xp2)
        j1 = find(x1==xp2(i));
        j1 = j1(end);
        [m, j2] = min(abs(f2-f1(j1)));
        xd2(i) = x2(j2);
    end;
    clear m j1 j2
    xd(i2) = xd2;
end;
clear xt1 xp2 xd2