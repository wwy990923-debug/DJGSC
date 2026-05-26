clc
clear
close all
warning off

%% Load data
filename1 = 'yaleborigin';
load(strcat(filename1, '.mat'))

addpath('D:\my_SRC-main\fista_lasso.m')

%% Output TXT file
filename = 'jgsc_yaleb_param_search';
fid = fopen(strcat(filename, '.txt'), 'a+');

dat = date;
fprintf(fid, '\r\n\r\n==============================\r\n');
fprintf(fid, 'Date: %s\r\n', dat);
fprintf(fid, 'Parameter search for GLSRC2\r\n');
fprintf(fid, '==============================\r\n');

%% Dataset settings
data = dataset;

nSet = 38;
nmax = 1;

number = 64;
number_train = 32;

%% Parameter search range
% lambda and gamma are searched from 0.0001 to 100 with 10x interval
lambda_list = 10.^(-4:2);   % 0.0001, 0.001, 0.01, 0.1, 1, 10, 100
gamma_list  = 10.^(-4:2);   % 0.0001, 0.001, 0.01, 0.1, 1, 10, 100

fprintf(fid, 'lambda_list = ');
fprintf(fid, '%g ', lambda_list);
fprintf(fid, '\r\n');

fprintf(fid, 'gamma_list = ');
fprintf(fid, '%g ', gamma_list);
fprintf(fid, '\r\n\r\n');

%% Record best result
best_acc = -inf;
best_lambda = NaN;
best_gamma = NaN;
best_time = NaN;

%% Main experiment loop
for k1 = 1:length(nSet)
    for n1 = 1:length(number_train)

        dataset = data(:, 1:nSet(k1) * number);

        fprintf(fid, '----------------------------------------\r\n');
        fprintf(fid, 'Class number: %d, Training samples per class: %d\r\n', ...
            nSet(k1), number_train(n1));
        fprintf(fid, '----------------------------------------\r\n');

        for l_id = 1:length(lambda_list)
            for g_id = 1:length(gamma_list)

                lambda1 = lambda_list(l_id);
                gamma1  = gamma_list(g_id);

                Bcc = zeros(nmax, 1);
                Tim = zeros(nmax, 1);

                fprintf(fid, '\r\nCurrent parameters: lambda=%g, gamma=%g\r\n', ...
                    lambda1, gamma1);

                for m5 = 1:nmax
                    tic;

                    %% Split training and testing samples
                    train_numbers = 1:number_train(n1);
                    test_numbers = 33:64;

                    S = [];
                    T = [];
                    S_label = [];
                    T_label = [];

                    for i = 1:nSet(k1)
                        for m1 = 1:length(train_numbers)
                            S = [S dataset(:, (i-1)*number + train_numbers(m1))];
                            S_label = [S_label i];
                        end

                        for m2 = 1:length(test_numbers)
                            T = [T dataset(:, (i-1)*number + test_numbers(m2))];
                            T_label = [T_label i];
                        end
                    end

                    %% Run GLSRC2 with current lambda and gamma
                    W = GLSRC2(S, T, number_train(n1), lambda1, gamma1);

                    %% Classification
                    [~, numRows] = size(W);

                    fi = zeros(numRows, nSet(k1));
                    for j = 1:numRows
                        for k = 1:nSet(k1)
                            fi(j, k) = norm(T(:, j) - S(:, S_label == k) * W(S_label == k, j));
                        end
                    end

                    [~, max_indices] = min(fi, [], 2);

                    correct_num = 0;
                    for j = 1:numRows
                        if T_label(1, j) == max_indices(j, 1)
                            correct_num = correct_num + 1;
                        end
                    end

                    acc = correct_num / numRows;
                    elapsed_time = toc;

                    Bcc(m5) = acc;
                    Tim(m5) = elapsed_time;

                    %% Print to command window
                    fprintf('lambda=%g, gamma=%g, run=%d, acc=%6.4f, time=%8.6f\n', ...
                        lambda1, gamma1, m5, acc, elapsed_time);

                    %% Write each run result to TXT
                    fprintf(fid, ['run=%d, lambda=%g, gamma=%g, class=%d, trainnumber=%d, ' ...
                        'acc=%6.4f, time=%8.6f\r\n'], ...
                        m5, lambda1, gamma1, nSet(k1), number_train(n1), acc, elapsed_time);
                end

                %% Summary for current parameter pair
                avgbcc = mean(Bcc);
                maxbcc = max(Bcc);
                minbcc = min(Bcc);
                stdbcc = std(Bcc);
                avgtim = mean(Tim);

                fprintf(fid, ['SUMMARY: lambda=%g, gamma=%g, class=%d, trainnumber=%d, ' ...
                    'avgbcc=%6.4f, minbcc=%6.4f, maxbcc=%6.4f, ' ...
                    'avgtim=%8.6f, stdbcc=%8.6f\r\n'], ...
                    lambda1, gamma1, nSet(k1), number_train(n1), ...
                    avgbcc, minbcc, maxbcc, avgtim, stdbcc);

                %% Update best result
                if avgbcc > best_acc
                    best_acc = avgbcc;
                    best_lambda = lambda1;
                    best_gamma = gamma1;
                    best_time = avgtim;
                end
            end
        end
    end
end

%% Write best result
fprintf(fid, '\r\n========================================\r\n');
fprintf(fid, 'BEST RESULT\r\n');
fprintf(fid, 'best_lambda=%g, best_gamma=%g, best_acc=%6.4f, best_time=%8.6f\r\n', ...
    best_lambda, best_gamma, best_acc, best_time);
fprintf(fid, '========================================\r\n');

fprintf('\nBest result: lambda=%g, gamma=%g, acc=%6.4f, time=%8.6f\n', ...
    best_lambda, best_gamma, best_acc, best_time);

%% Close TXT file
fclose(fid);
