clc
clear
close all
warning off


filename1 = 'yaleborigin';
load(strcat(filename1,'.mat'))
addpath('D:\my_SRC-main\fista_lasso.m')
filename = 'djgsc_yaleb';
fid = fopen(strcat(filename,'.txt'),'a+');
dat = date;
fprintf(fid,strcat('\r\n\r\n', dat,'\r\n'));

data = dataset;
method = 2;
nSet = 38;
nmax = 10;
number = 64;
number_train = 32;
alpha = 0.1;
for k1 = 1:length(nSet)
    for n1 = 1:length(number_train)
        dataset = data(:,1:nSet(k1)*number);
        Bcc = zeros(nmax,1);
        Acc = zeros(nmax,1);
        Tim = zeros(nmax,1);
        for q5 = 1:length(alpha)
            alpha1 = alpha(q5);
            for m5 = 1: nmax
                tic;
                random_permutation = randperm(64);
                train_numbers = 1:number_train(n1);
                test_numbers = 1:32;
                test_numbers = 33:64;
                S = [];
                T = [];
                S_label = [];
                T_label = [];
                for i = 1:nSet(k1)
                    for m1 = 1:length(train_numbers)
                        S = [S dataset(:,(i-1)*64+train_numbers(m1))];
                        S_label = [S_label i];
                    end
                    for m2 = 1:length(test_numbers)
                        T = [T dataset(:,(i-1)*64+test_numbers(m2))];
                        T_label = [T_label i];
                    end

                end
           
                W = DJGSC(S,T,number_train(n1),0.01,10);
                [~, numRows] = size(W);
                fi = zeros(numRows,nSet(k1));
                for j = 1:numRows
                    for k = 1:nSet(k1)
                        fi(j,k) = norm(T(:,j)-S(:,S_label == k)*W(S_label == k,j));
                    end
                end
                [max_values, max_indices] = min(fi, [], 2);
                abb = 0;
                for j = 1:numRows
                    if T_label(1,j) == max_indices(j,1)
                        abb = abb+1;
                    end
                end
                bcc = abb/numRows;
                confMat = confusionmat(T_label, max_indices');
                toc;
                elapsed_time = toc;
                tic
                Tim(m5) = elapsed_time;
                Bcc(m5) = bcc;
                fprintf('acc=%f ', bcc);
                if m5 == nmax
                    avgbcc = mean(Bcc);
                    maxbcc = max(Bcc);
                    minbcc = min(Bcc);
                    stdbcc = std(Bcc);
                    avgtim = mean(Tim);
                    fprintf(fid,['#n=%6.4f, class:%6.1f, trainnumber:%6.1f, avgbcc:%6.4f, minbcc:%6.4f, maxbcc:%6.4f, ' ...
                        ', avgtim:%8.6f, stdbcc:%8.6f,alpha:%8.6f\n'],m5,nSet(k1),number_train(n1),avgbcc,minbcc,maxbcc,avgtim,stdbcc,alpha1);
                end
            end
        end
    end
end
