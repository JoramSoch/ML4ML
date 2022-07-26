function [h, p, rr, stats] = ML_SVC_chi2(SVC1, SVC2, alpha)
% _
% McNemar's Test for Comparing two Support Vector Classifications
% FORMAT [h, p, rr, stats] = ML_SVC_chi2(SVC1, SVC2, alpha)
% 
%     SVC1  - a structure specifying the first SVC
%     SVC2  - a structure specifying the second SVC
%     alpha - a scalar, the significance level for McNemar's test
% 
%     h     - an subs x perm matrix of hypotheses (subs = subsamples,
%     p     - an subs x perm matrix of p-values    perm = permutations)
%     rr    - an subs x perm matrix of relative risks of misclassification
%             (for first relative to second classifyer)
%     stats - a structure specifying test statistics
%     o chi2  - an subs x perm matrix of chi-square test statistics
%     o df    - a scalar, the number of degrees of freedom
%     o cv    - a scalar, the critical value of the test
% 
% FORMAT [h, p, rr, stats] = ML_SVC_chi2(SVC1, SVC2, alpha) performs McNemar's
% test between support vector machines SVC1 and SVC2, using significance
% level alpha, and returns hypothesis h (1: reject, 0: no reject), p-value
% p, relative risk rr and test statistics stats.
% 
% McNemar's test for two classification algorithms [1] assesses the null
% hypothesis that rates of misclassification are equal between classifiers.
% Consequently, if the null hypothesis is rejected, one algorithms has
% fewer misclassifications than the other.
% 
% If an SVC analysis has been run with (i) subsampling or (ii) permutations,
% the test is performed for each subsample-permutation combination
% indepently and results for all subsample-permutation combinations are
% returned as matrices. Sample size, number of subsamples and permutations
% must be equal between compared SVCs.
% 
% References:
% [1] Dietterich TG (1998). Approximate Statistical Tests for Comparing
%     Supervised Classification Learning Algorithms. Neural Computation,
%     vol. 10, iss. 7, pp. 1895-1923; DOI: 10.1162/089976698300017197.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 01/07/2022, 14:21
%  Last edit: 26/07/2022, 12:03


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(alpha), alpha = 0.05; end;

% Get permutations and subsamples
%-------------------------------------------------------------------------%
perm = SVC1.pars.perm;
subs = SVC1.pars.subs;
if subs == 0, subs = 1; end;
% Note: The algorithm assumes that the number of permutations
% and number of subsamples was equal between the two SVCs.

% Preallocate test results
%-------------------------------------------------------------------------%
rr         = zeros(subs,perm);
stats.chi2 = zeros(subs,perm);

% Perform statistical test
%-------------------------------------------------------------------------%
for i = 1:subs
    for j = 1:perm
        
        % Get true and predicted labels
        %-----------------------------------------------------------------%
        xt  = SVC1.pred.xt(:,j,i);
        xp1 = SVC1.pred.xp(:,j,i);
        xp2 = SVC2.pred.xp(:,j,i);
        % Note: True labels are always taken from the first classifier and
        % assumed to be identical in the second classifier. When the number
        % of subjects and number of classes is equal between classifiers,
        % all subsamples and permutations are guaranteed to be equivalent.
        
        % Generate confusion matrix
        %-----------------------------------------------------------------%
        CM = [sum(xp1~=xt & xp2~=xt), sum(xp1~=xt & xp2==xt);
              sum(xp1==xt & xp2~=xt), sum(xp1==xt & xp2==xt)];
        
        % Calculate test statistic
        %-----------------------------------------------------------------%
        stats.chi2(i,j) = (abs(CM(1,2)-CM(2,1))-1)^2 / (CM(1,2)+CM(2,1));
        
        % Calculate relative risk
        %-----------------------------------------------------------------%
        rr(i,j) = (CM(1,1)+CM(1,2)) / (CM(1,1)+CM(2,1));
        
    end;
end;

% Obtain test results
%-------------------------------------------------------------------------%
stats.df = 1;
stats.cv = chi2inv(1-alpha, stats.df);
h = stats.chi2 > stats.cv;
p = 1 - chi2cdf(stats.chi2, stats.df);