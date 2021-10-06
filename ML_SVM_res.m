function ML_SVM_res(SVM, type)
% _
% Results Display for Cross-Validated Support Vector Machines
% FORMAT ML_SVM_res(SVM, type)
%     SVM  - a structure specifying a calibrated SVM
%     type - the type of results to be displayed (see below)
% 
% FORMAT ML_SVM_res(SVM, type) displays results upon SVM analysis.
% 
% FORMAT ML_SVM_res; uses "spm_select" to choose a saved SVM.mat and
% uses "spm_input" to select the type of results display.
% 
% The input variable "data" has to be one out of four strings:
% - 'data': displays class labels/target values and feature matrix;
% - 'pars': displays cross-validation structure and hyper-parameter value;
% - 'pred': displays true and predicted class labels/target values;
% - 'perf': displays predictive performance and permutation distribution.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 25/08/2021, 13:04
%  Last edit: 06/10/2021, 10:33


% Get SVM.mat if necessary
%-------------------------------------------------------------------------%
if nargin == 0
    SVM_mat = spm_select(1,'^SVM\.mat$','Select SVM.mat!');
    load(SVM_mat);
    ML_SVM_res(SVM);
    return
end;

% Select type of display
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(type)
    types = {'input data', 'SVM parameters', 'SVM predictions', 'predictive performance'};
    ind   = spm_input('Select type of results display', 1, 'm', types, [1:numel(types)]);
    types = {'data', 'pars', 'pred', 'perf'};
    type  = types{ind};
end;
clear ind

% Get SVM basic information
%-------------------------------------------------------------------------%
SVC   = SVM.is_SVC;
[n,v] = size(SVM.data.Y);
if isfield(SVM.pars,'perm'), perm = SVM.pars.perm;
else,                        perm = 1; end;
if isfield(SVM.pars,'subs'), subs = SVM.pars.subs;
else,                        subs = 0; end;

% Type == 'data'
%-------------------------------------------------------------------------%
if strcmp(type,'data')
    figure('Name', 'SVM: data', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
    % class labels / target values
    subplot(1,10,1);
    imagesc(SVM.data.x);
    if SVC
        caxis([1 SVM.data.m]);
        cb = colorbar;
        set(cb, 'Ticks', [1:SVM.data.m], 'TickLabels', num2cell(1:SVM.data.m));
    else
        caxis([min(SVM.data.x)-1/10*range(SVM.data.x), max(SVM.data.x)+1/10*range(SVM.data.x)]);
        colorbar;
    end;
    set(gca,'XTick',[1],'XTickLabel',{' '});
    if SVC, NpC = ['[', sprintf('%d, ', SVM.data.N(1:end-1)), sprintf('%d', SVM.data.N(end)), ']']; end;
    if SVC, ylabel(sprintf('%d classes, points per class: %s', SVM.data.m, NpC), 'FontSize', 16);
    else,   ylabel(sprintf('values between %1.2f and %1.2f', min(SVM.data.x), max(SVM.data.x)), 'FontSize', 16); end;
    if SVC, title(sprintf('%d class labels', n), 'FontSize', 16);
    else,   title(sprintf('%d target values', n), 'FontSize', 16); end;
    % feature matrix
    subplot(1,10,[2:10]);
    imagesc(SVM.data.Y);
    caxis([-max(max(abs(SVM.data.Y))), +max(max(abs(SVM.data.Y)))]);
    colorbar;
    xlabel('feature', 'FontSize', 12);
    ylabel('data point', 'FontSize', 12);
    title(sprintf('%d x %d feature matrix', n, v), 'FontSize', 16);
end;

% Type == 'pars'
%-------------------------------------------------------------------------%
if strcmp(type,'pars')
    figure('Name', 'SVM: parameters', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
    % cross-validation matrix
    CV = SVM.pars.CV(:,:,1);
    k  = size(CV,2);
    imagesc(CV);
    caxis([1 2]);
    cb = colorbar;
    set(cb, 'Ticks', [1:2], 'TickLabels', num2cell(1:2));
    ylabel('cross-validation set (1 = training, 2 = test)', 'FontSize', 16);
    if SVC, xlabel(sprintf('cross-validation fold (linear SVC, k = %d, C = %g)', k, SVM.pars.C), 'FontSize', 16);
    else,   xlabel(sprintf('cross-validation fold (linear SVR, k = %d, C = %g)', k, SVM.pars.C), 'FontSize', 16); end;
    if subs==0, title(sprintf('%d x %d cross-validation matrix', n, k), 'FontSize' ,16);
    else,       title(sprintf('%d x %d cross-validation matrix (subsample 1 out of %d)', size(CV,1), k, subs), 'FontSize' ,16); end;
end;

% Type == 'pred'
%-------------------------------------------------------------------------%
if strcmp(type,'pred')
    figure('Name', 'SVM: predictions', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
    % subsamples
    if subs > 0
        subplot(2,2,1);
        is = SVM.pred.is;
        ns = min([10, size(is,2)]);
        imagesc(is(:,1:ns));
        caxis([1 n]);
        colorbar;
        xlabel('subsample', 'FontSize', 12);
        ylabel('data point', 'FontSize', 12);
        title(sprintf('subsamples (1-%d out of %d)', ns, subs), 'FontSize', 16);
    end;
    % permutations
    if perm > 1
        subplot(2,2,3);
        ip = SVM.pred.ip(:,:,1);
        np = min([10, size(ip,2)]);
        imagesc(ip(:,1:np));
        caxis([1 n]);
        colorbar;
        if subs==0, xlabel('permutation', 'FontSize', 12);
        else,       xlabel('permutation (subsample 1)', 'FontSize', 12); end;
        ylabel('data point', 'FontSize', 12);
        title(sprintf('permutations (1-%d out of %d)', np, perm), 'FontSize', 16);
    end;
    % true labels
    subplot(2,2,2);
    xt = SVM.pred.xt(:,:,1);
    np = min([10, size(xt,2)]);
    imagesc(xt(:,1:np));
    if SVC
        caxis([1 SVM.data.m]);
        cb = colorbar;
        set(cb, 'Ticks', [1:SVM.data.m], 'TickLabels', num2cell(1:SVM.data.m));
    else
        caxis([min(xt(:,1))-1/10*range(xt(:,1)), max(xt(:,1))+1/10*range(xt(:,1))]);
        colorbar;
    end;
    if subs==0, xlabel('permutation', 'FontSize', 12);
    else,       xlabel('permutation (subsample 1)', 'FontSize', 12); end;
    ylabel('data point', 'FontSize', 12);
    if SVC, title('true class labels', 'FontSize', 16);
    else,   title('true target values', 'FontSize', 16); end;
    % predicted labels
    subplot(2,2,4);
    xp = SVM.pred.xp(:,:,1);
    np = min([10, size(xp,2)]);
    imagesc(xp(:,1:np));
    if SVC
        caxis([1 SVM.data.m]);
        cb = colorbar;
        set(cb, 'Ticks', [1:SVM.data.m], 'TickLabels', num2cell(1:SVM.data.m));
    else
        caxis([min(xp(:,1))-1/10*range(xp(:,1)), max(xp(:,1))+1/10*range(xp(:,1))]);
        colorbar;
    end;
    if subs==0, xlabel('permutation', 'FontSize', 12);
    else,       xlabel('permutation (subsample 1)', 'FontSize', 12); end;
    ylabel('data point', 'FontSize', 12);
    if SVC, title('predicted class labels', 'FontSize', 16);
    else,   title('predicted target values', 'FontSize', 16); end;
end;

% Type == 'perf'
%-------------------------------------------------------------------------%
if strcmp(type,'perf')
    figure('Name', 'SVM: performance', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
    % support vector classification
    if SVC
        % decoding accuracies
        subplot(2,2,1); hold on;
        Acc = [mean(SVM.perf.DA(1,1,:),3), mean(SVM.perf.BA(1,1,:),3), mean(SVM.perf.CA(:,1,:),3)'];
        CIs = [mean(SVM.perf.DA_CI,3)',    mean(SVM.perf.BA_CI,3)',    mean(SVM.perf.CA_CI,3)'];
        Accs= [reshape(SVM.perf.DA(1,1,:),[size(SVM.perf.DA,3) 1]), ...
               reshape(SVM.perf.BA(1,1,:),[size(SVM.perf.BA,3) 1]), ...
               reshape(SVM.perf.CA(:,1,:),[SVM.data.m, size(SVM.perf.CA,3)])'];
        bar(1, Acc(1), 'r');
        bar(2, Acc(2), 'g');
        bar([3:numel(Acc)], Acc(3:end), 'b');
        errorbar([1:numel(Acc)], Acc, Acc-CIs(1,:), CIs(2,:)-Acc, '.k', 'LineWidth', 2, 'CapSize', 20);
        plot([0, numel(Acc)+1], [1/SVM.data.m, 1/SVM.data.m], ':k', 'LineWidth', 2);
        if subs > 0
            rng(size(Accs,1));
            for h = 1:size(Accs,2)
                plot(h+3/4*(rand(size(Accs,1),1)-1/2), Accs(:,h), '.k', 'LineWidth', 2, 'MarkerSize', 10);
            end;
        end;
        axis([0, numel(Acc)+1, 0, 1]);
        set(gca,'Box','On');
        set(gca,'XTick',[1:numel(Acc)],'XTickLabel',[{'DA', 'BA'}, cellstr(num2str([1:SVM.data.m]'))']);
        legend('decoding accuracy', 'balanced accuracy', 'class accuracies', '90% confidence intervals', 'chance level', 'Location', 'South');
        xlabel('measure', 'FontSize', 12);
        ylabel('accuracy', 'FontSize', 12);
        title('unpermuted data: predictive performance', 'FontSize', 16);
        % confusion matrix
        subplot(2,2,3);
        CM = mean(SVM.perf.CM,3);
        imagesc(CM);
        caxis([0 1]);
        colorbar;
        axis xy;
        set(gca,'XTick',[1:SVM.data.m]);
        set(gca,'YTick',[1:SVM.data.m]);
        xlabel('true class', 'FontSize', 12);
        ylabel('predicted class', 'FontSize', 12);
        title('unpermuted data: confusion matrix', 'FontSize', 16);
        for c1 = 1:SVM.data.m
            for c2 = 1:SVM.data.m
                text(c1, c2, sprintf('%0.4f', CM(c2,c1)), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
            end;
        end;
        % permutation testing (DA)
        subplot(2,2,2); hold on;
        DAs = reshape(SVM.perf.DA(1,:,:),[1 size(SVM.perf.DA,3)*perm]);
        DA1 = reshape(SVM.perf.DA(1,1,:),[1 size(SVM.perf.DA,3)]);
        dx  = 0.01;
        x   = [(0+dx/2):dx:(1-dx/2)];
        n   = hist(DAs, x);
        if subs > 0, n = n./subs; end;
        bar(x, n, 'b');
        plot(DA1, 0, 'xr', 'LineWidth', 2, 'MarkerSize', 10);
        axis([0, 1, 0, (11/10)*max(n)]);
        if mean(SVM.perf.DA_pp) < 0.001
            text(0.95, (21/20)*max(n), 'mean p < 0.001', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Middle');
        else
            text(0.95, (21/20)*max(n), sprintf('mean p = %0.3f', mean(SVM.perf.DA_pp)), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Middle');
        end;
        set(gca,'Box','On');
        legend('permutation distribution', 'observed accuracies', 'Location', 'NorthWest');
        xlabel('decoding accuracy', 'FontSize', 12);
        ylabel('number of permutations', 'FontSize', 12);
        title('permutation testing: decoding accuracy', 'FontSize', 16);
        % permutation testing (BA)
        subplot(2,2,4); hold on;
        BAs = reshape(SVM.perf.BA(1,:,:),[1 size(SVM.perf.BA,3)*perm]);
        BA1 = reshape(SVM.perf.BA(1,1,:),[1 size(SVM.perf.BA,3)]);
        dx  = 0.01;
        x   = [(0+dx/2):dx:(1-dx/2)];
        n   = hist(BAs, x);
        if subs > 0, n = n./subs; end;
        bar(x, n, 'b');
        plot(BA1, 0, 'xr', 'LineWidth', 2, 'MarkerSize', 10);
        axis([0, 1, 0, (11/10)*max(n)]);
        if mean(SVM.perf.BA_pp) < 0.001
            text(0.95, (21/20)*max(n), 'mean p < 0.001', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Middle');
        else
            text(0.95, (21/20)*max(n), sprintf('mean p = %0.3f', mean(SVM.perf.BA_pp)), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Middle');
        end;
        set(gca,'Box','On');
        legend('permutation distribution', 'observed accuracies', 'Location', 'NorthWest');
        xlabel('balanced accuracy', 'FontSize', 12);
        ylabel('number of permutations', 'FontSize', 12);
        title('permutation testing: balanced accuracy', 'FontSize', 16);
    % support vector regression
    else
        % predictive correlation
        subplot(2,2,1); hold on;
        Corr = [SVM.perf.r(1)];
        CI   = [SVM.perf.r_CI];
        bar(1, Corr(1), 'r');
        errorbar(1, Corr, Corr-CI(1), CI(2)-Corr, '.k', 'LineWidth', 2, 'CapSize', 15);
        plot([0, 3], [0, 0], '-k', 'LineWidth', 2);
        axis([0, 3, -1, +1]);
        set(gca,'Box','On');
        set(gca,'XTick',[1:2],'XTickLabel',{'r', 'R^2, MAE, MSE'});
        legend('predictive correlation', '90% confidence intervals', 'chance level', 'Location', 'SouthEast');
        xlabel('measure', 'FontSize', 12);
        ylabel('precision', 'FontSize', 12);
        title('unpermuted data: predictive performance', 'FontSize', 16);
        if Corr > 0
            text(2, Corr, sprintf('R^2 = %0.2f\nMAE = %1.2f\nMSE = %1.2f', SVM.perf.R2, SVM.perf.MAE, SVM.perf.MSE), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Top');
        else
            text(2, Corr, sprintf('R^2 = %0.2f\nMAE = %1.2f\nMSE = %1.2f', SVM.perf.R2, SVM.perf.MAE, SVM.perf.MSE), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom');
        end;
        % predicted vs. true values
        subplot(2,2,3); hold on;
        xr = [min(SVM.data.x)-1/10*range(SVM.data.x), max(SVM.data.x)+1/10*range(SVM.data.x)];
        yr = SVM.perf.m*xr + SVM.perf.n;
        plot(SVM.pred.xt(:,1), SVM.pred.xp(:,1), '.b', 'LineWidth', 2, 'MarkerSize', 10);
        plot(xr, yr, '-k', 'LineWidth', 2);
        axis([xr, min(SVM.pred.xp(:,1))-1/10*range(SVM.pred.xp(:,1)), max(SVM.pred.xp(:,1))+1/10*range(SVM.pred.xp(:,1))]);
        set(gca,'Box','On');
        legend('target values', 'regression line', 'Location', 'South');
        xlabel('true target values', 'FontSize', 12);
        ylabel('predicted target values', 'FontSize', 12);
        title('unpermuted data: predicted vs. true', 'FontSize', 16);
        text(mean(xr), max(SVM.pred.xp(:,1))+1/20*range(SVM.pred.xp(:,1)), sprintf('y = %1.2f x + %1.2f', SVM.perf.m, SVM.perf.n), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
        % permutation testing (r)
        subplot(2,2,2); hold on;
        rs = SVM.perf.r;
        dx = 0.02;
        x  = [(-1+dx/2):dx:(+1-dx/2)];
        n  = hist(rs, x);
        bar(x, n, 'b');
        plot(Corr, 0, 'xr', 'LineWidth', 2, 'MarkerSize', 10);
        plot([SVM.perf.r_cv(1), SVM.perf.r_cv(1)], [0, (11/10)*max(n)], ':k', 'LineWidth', 2);
        plot([SVM.perf.r_cv(2), SVM.perf.r_cv(2)], [0, (11/10)*max(n)], ':k', 'LineWidth', 2);
        axis([-1, +1, 0, (11/10)*max(n)]);
        if SVM.perf.r_pp < 0.001
            text(0.9, (21/20)*max(n), 'p < 0.001', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Middle');
        else
            text(0.9, (21/20)*max(n), sprintf('p = %0.3f', SVM.perf.r_pp), 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Middle');
        end;
        set(gca,'Box','On');
        legend('permutation distribution', 'observed accuracy', '90% critical values', 'Location', 'NorthWest');
        xlabel('predictive correlation', 'FontSize', 12);
        ylabel('number of permutations', 'FontSize', 12);
        title('permutation testing: predictive correlation', 'FontSize', 16);
    end;
end;