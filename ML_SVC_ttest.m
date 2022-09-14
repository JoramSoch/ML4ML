function [h, p, ci, stats] = ML_SVC_ttest(SVC1, SVC2, alpha)
% _
% Paired T-Test for Comparing two Support Vector Classifications
% FORMAT [h, p, ci, stats] = ML_SVC_ttest(SVC1, SVC2, alpha)
% 
%     SVC1  - a structure specifying the first SVC
%     SVC2  - a structure specifying the second SVC
%     alpha - a scalar, the significance level for the t-test
% 
%     h     - a logical indicating rejection of the null hypothesis
%     p     - a scalar, the the p-value of the paired t-test
%     ci    - a 2 x 1 vector, the confidence interval
%     stats - a structure specifying test statistics
%     o tstat - a scalar, the value of the t-statistic
%     o df    - a scalar, the number of degrees of freedom
%     o cv    - a scalar, the critical value of the test
%     o R     - an s x s matrix, the subsample correlation structure
% 
% FORMAT [h, p, ci, stats] = ML_SVC_ttest(SVC1, SVC2, alpha) performs a
% paired t-test across the subsamples for two support vector classifications
% SVC1 and SVC2, using the permutations for forming the correlation matrix;
% it returns hypothesis h (1: reject, 0: no reject), p-value p, confidence
% interval ci and test statistics stats.
% 
% The procedure requires that an SVC analysis has been run with (i) a number
% of subsamples large enough for inferential statistics and (ii) a number
% of permutations larger than the number of samples. Suitable values would
% e.g. be subs = 30 subsamples and perm = 1000 permutations (see "ML_SVC"
% for more information). Sample size, number of subsamples and permutations
% must be equal between compared SVCs.
% 
% The test assumes that the subsamples are not independent, but that their
% dependence can be estimated. Subsample inter-correlation is quantified by
% calculating the correlation matrix between subsamples across all
% permutations. This is an s x s matrix which is then integrated into the
% linear model [1,2,3] testing for non-zero performance difference between
% the two classification algorithms.
% 
% This routine performs a one-sided t-test against the null hypothesis
% that the average decoding accuracy difference between SVC1 and SVC2 is
% larger than zero, i.e. that SVC1 performs better than SVC2. For a two-
% sided test, recalculate the p-value as p2 = 2 * min([p1, 1-p1]).
% 
% References:
% [1] Soch J (2020). Contrast-Based Inference for Classical General Linear Model;
%     URL: https://github.com/JoramSoch/MACS/blob/master/ME_GLM_con.m
% [2] Soch J (2021). Plot Contrast with Confidence Intervals;
%     URL: https://github.com/JoramSoch/spm_helper/blob/master/spm_plot_con_CI.m
% [3] Soch J (2022). Multi-Modal Analyses using Voxel-Wise General Linear Models;
%     URL: https://github.com/JoramSoch/MMA/blob/main/mma_mma.m
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 26/07/2022, 09:19
%  Last edit: 26/07/2022, 12:28


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(alpha), alpha = 0.05; end;

% Extract decoding accuraices
%-------------------------------------------------------------------------%
Y1 = squeeze(SVC1.perf.DA);     % permutations x subsamples
Y2 = squeeze(SVC2.perf.DA);     % permutations x subsamples
y  = (Y1(1,:)-Y2(1,:))';        % performance differences
X  = ones(size(y));             % one-sample paired t-test
V  = corr(Y1-Y2);               % subsample correlation structure
c  = 1;                         % test for the mean difference

% Perform statistical test
%-------------------------------------------------------------------------%
[h, p, stats] = ME_GLM_con(y, X, V, c, 't', alpha);
stats.cv = tinv(1-alpha, stats.df);
stats.R  = V;
h = stats.tstat > stats.cv;
p = 1 - tcdf(stats.tstat, stats.df);

% Calculate confidence interval
%-------------------------------------------------------------------------%
[ym, s2] = ME_GLM(y, X, V);
covB = inv(X'*inv(V)*X);
se   = sqrt(s2 * c'*covB*c);
ci   = se * norminv(1-alpha/2, 0, 1);
ci   = [ym - ci; ym + ci];