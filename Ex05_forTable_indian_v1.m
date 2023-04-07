%% Loop for alpha

clc
tic
close all
clear
data_path = './data/Indian_pines_corrected.mat';
label_path = './data/Indian_pines_gt.mat';
addpath("function/");
load(data_path);
data = indian_pines_corrected;
clear indian_pines_corrected;

load(label_path);
label = indian_pines_gt;
clear indian_pines_gt;
c = length(unique(label));


%% resize data to a small size

% hh2=10;
% [ww,hh,bb]=size(data);
% ww2=floor(ww/hh*hh2);
% data2=zeros(ww2,hh2,bb);
% for i=1:bb
%     a=data(:,:,i);
%     maxv=max(max(a));
%     minv=min(min(a));
%     a = floor(a-minv).*(255/maxv-minv);
%     data2(:,:,i) = imresize(a,[ww2,hh2]);
% end
% data= double(data2);
% clear data2 ww hh a bb minv maxv

%%
hh2 = 10;%20;
ww2 = 20;

%% loop
beta=[1e-3 1e-2 1e-1 1 1e1 1e2 1e3];
selectedBands = 5:5:50;
alpha=1e-3;
acc_s = zeros(length(beta), length(selectedBands));
acc_k = acc_s;
acc_l = acc_s;
acc_std_s = zeros(length(beta), length(selectedBands));
acc_std_k = acc_s;
acc_std_l = acc_s;

kappa_s = zeros(length(beta), length(selectedBands));
kappa_k = acc_s;
kappa_l = acc_s;
kappa_std_s = zeros(length(beta), length(selectedBands));
kappa_std_k = acc_s;
kappa_std_l = acc_s;
for m=1:length(beta)
%% preprocess
%pyenv('Version', '/Users/u/opt/anaconda3/bin/python3')
%pyenv('ExecutionMode','OutOfProcess')
w = hh2;%10;
h = ww2;%10;
[X, y] = preprocess2(data, label, w, h);
disp('prepared...');
% index = SPCA_AMGL(X, 1, 1, 1, 15);
%X = NormalizeFea(X, 0);
[index, Z, S, W, obj, score] = MultiGraphBS_Botswana(X, beta(m), alpha, c, 10);
disp('compute band selection index done.');


%% Ex01
tic
for i=1:length(selectedBands)
    sorted_index = py.numpy.array(floor(index(1:selectedBands(i))-1));%change to python index: -1
    % Call the evaluate_band_selection function in Python
    %acc_s, acc_k, acc_l = evaluate_band_selection(sorted_index, pattern,data_path,label_path)
    [acc] = pyrunfile("evaluate2.py", "acc", ...
        sorted_index = sorted_index, pattern = 0, ...
        data_path = data_path, label_path = label_path);
    % Convert Python data type back to Matlab type
    acc=double(acc);
    acc_s(m,i) = double(acc(1));
    acc_k(m,i) = double(acc(2));
    acc_l(m,i) = double(acc(3));
    acc_std_s(m,i) = double(acc(4));
    acc_std_k(m,i) = double(acc(5));
    acc_std_l(m,i) = double(acc(6));

    kappa_s(m,i) = double(acc(7));
    kappa_k(m,i) = double(acc(8));
    kappa_l(m,i) = double(acc(9));
    kappa_std_s(m,i) = double(acc(10));
    kappa_std_k(m,i) = double(acc(11));
    kappa_std_l(m,i) = double(acc(12));
end
disp([num2str(m) ' / ' num2str(7) '...'])
%disp('evaluate done.');
end
%%
disp('------------------------------------------')
disp(['SVM(AOA): ' num2str(max(max(acc_s)))])
disp(['SVM(Kappa): ' num2str(max(max(kappa_s)))])
disp('------------------------------------------')
disp(['KNN(AOA): ' num2str(max(max(acc_k)))])
disp(['KNN(Kappa): ' num2str(max(max(kappa_k)))])
disp('------------------------------------------')
disp(['LDA(AOA): ' num2str(max(max(acc_l)))])
disp(['LDA(Kappa): ' num2str(max(max(kappa_l)))])
disp('------------------------------------------')
disp('all sensitivity experiments done.');
%%
toc
