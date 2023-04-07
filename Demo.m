% 20230107 by Mengbo You 

%% Load dataset path
clc
close all
clear
data_path = './data/Indian_pines_corrected.mat';
label_path = './data/Indian_pines_gt.mat';
addpath("functions\");

load(data_path);
data = indian_pines_corrected;
clear indian_pines_corrected;

load(label_path);
label = indian_pines_gt;
clear indian_pines_gt;
c = length(unique(label));

%% Prepare
%======================= Parameters ================================
flag_load_rst = true;% Load band selection result or "do band selection"
alpha=1e-6;
beta=1e-3;
selectedBands = 5:5:50;
%===================================================================
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

%% Load band selection result or do band selection
if flag_load_rst
    load('.\rst\band_selection_rst_indian_pines.mat','index');
else
    w = 10;% sample size
    h = 10;% sample size
    [X, y] = preprocess(data, label, w, h);
    disp('prepared...');
    X = NormalizeFea(X, 0);
    tic;
    [index, Z, S, W] = MultiGraphBS_noGPU(X, beta, alpha, c);
    disp('compute band selection index done.');
    toc;
end

%% Evaluate
for i=1:length(selectedBands)
    sorted_index = py.numpy.array(floor(index(1:selectedBands(i))-1));
    [acc_s0, acc_k0, acc_l0, accstd_s, accstd_k, accstd_l] = pyrunfile("evaluate.py", ["acc_s", "acc_k", "acc_l", "accstd_s", "accstd_k", "accstd_l"], ...
        sorted_index = sorted_index, pattern = 0, ...
        data_path = data_path, label_path = label_path);
    acc_s(i) = double(acc_s0);
    acc_k(i) = double(acc_k0);
    acc_l(i) = double(acc_l0);
    acc_std_s(i) = double(accstd_s);
    acc_std_k(i) = double(accstd_k);
    acc_std_l(i) = double(accstd_l);

    [acc_s0, acc_k0, acc_l0, accstd_s, accstd_k, accstd_l] = pyrunfile("evaluate.py", ["acc_s", "acc_k", "acc_l", "accstd_s", "accstd_k", "accstd_l"], ...
        sorted_index = sorted_index, pattern = 1, ...
        data_path = data_path, label_path = label_path);
    kappa_s(i) = double(acc_s0);
    kappa_k(i) = double(acc_k0);
    kappa_l(i) = double(acc_l0);
    kappa_std_s(i) = double(accstd_s);
    kappa_std_k(i) = double(accstd_k);
    kappa_std_l(i) = double(accstd_l);
end

%% Show result
disp('------------------------------------------')
disp(['SVM(OA): ' num2str(max(acc_s))])
disp(['KNN(OA): ' num2str(max(acc_k))])
disp(['LDA(OA): ' num2str(max(acc_l))])
disp('------------------------------------------')
disp(['SVM(Kappa): ' num2str(max(kappa_s))])
disp(['KNN(Kappa): ' num2str(max(kappa_k))])
disp(['LDA(Kappa): ' num2str(max(kappa_l))])
disp('------------------------------------------')
disp('evaluate done.');


%% Output example
% ------------------------------------------
% SVM(OA): 80.24
% KNN(OA): 74.8
% LDA(OA): 70.09
% ------------------------------------------
% SVM(Kappa): 77.02
% KNN(Kappa): 71.41
% LDA(Kappa): 66.14
% ------------------------------------------
% evaluate done.
