import numpy as np
import math
from scipy.io import loadmat
from sklearn.preprocessing import normalize
import joblib
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


#pattern=0, acc
#pattern=1, kappa
def evaluate_band_selection(sorted_index, pattern,data_path,label_path):
    data_mat, label_mat = preprocess_normal_std(data_path, label_path)
    acc_s = 0
    acc_k = 0
    acc_l = 0
    accstd_s = 0
    accstd_k = 0
    accstd_l = 0
    iteration = 1
    for i in range(iteration):
        data_mat_new = data_mat[:,sorted_index.astype(int)]
        num = 10
        a = np.zeros(num)
        b = np.zeros(num)
        c = np.zeros(num)
        for i in range(num):
            a[i] = eval_band_SVM(data_mat_new, label_mat, pattern)
            b[i] = eval_band_KNN(data_mat_new, label_mat, pattern)
            c[i] = eval_band_LDA(data_mat_new, label_mat, pattern)
        mean_svm = np.mean(a)
        mean_knn = np.mean(b)
        mean_lda = np.mean(c)
        std_svm = np.std(a)
        std_knn = np.std(b)
        std_lda = np.std(c)
        acc_s += mean_svm
        acc_k += mean_knn
        acc_l += mean_lda
        accstd_s += std_svm
        accstd_k += std_knn
        accstd_l += std_lda
    acc_s = round(acc_s/iteration, 2)
    acc_k = round(acc_k/iteration, 2)
    acc_l = round(acc_l/iteration, 2)
    accstd_s = round(accstd_s/iteration, 2)
    accstd_k = round(accstd_k/iteration, 2)
    accstd_l = round(accstd_l/iteration, 2)
    if pattern == 0:
        print("acc_svm:{}, acc_knn:{}, acc_lda:{}".format(acc_s, acc_k, acc_l))
    elif pattern == 1:
        print("kappa_svm:{}, kappa_knn:{}, kappa_lda:{}".format(acc_s, acc_k, acc_l))
    
    return acc_s, acc_k, acc_l, accstd_s, accstd_k, accstd_l

def preprocess_normal_std(data_path, label_path):
    data_mat = loadmat(data_path)
    label_mat = loadmat(label_path)

    data_index = list(data_mat)[-1]
    data_mat = data_mat[data_index]
    label_index = list(label_mat)[-1]
    label_mat = label_mat[label_index]
    
    m, n, b = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]

    data_mat = data_mat.reshape(m * n, b)
    label_mat = label_mat.reshape(m * n)

    index_nozero = np.where(label_mat != 0)[0]
    data_mat = data_mat[index_nozero]
    label_mat = label_mat[index_nozero]
    
    data_mat = normalize(data_mat)

    return data_mat, label_mat  
def eval_band_SVM(data_mat, labels_mat, pattern):
    
    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    clf = SVC(kernel='rbf', C=10000)
    clf.fit(data_train, labels_train)
    labels_pred = clf.predict(data_test)
    if pattern == 0:
        accuracy = metrics.accuracy_score(labels_test, labels_pred) * 100
    elif pattern == 1:
        accuracy = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(clf, 'work/model/Indian_pines_svm_model.m')

    return accuracy

def eval_band_KNN(data_mat, labels_mat, pattern):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
    neigh.fit(data_train, labels_train)
    labels_pred = neigh.predict(data_test)
    if pattern == 0:
        accuracy = metrics.accuracy_score(labels_test, labels_pred) * 100
    elif pattern == 1:
        accuracy = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_knn_model.m')

    return accuracy

def eval_band_LDA(data_mat, labels_mat, pattern):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    clf = LinearDiscriminantAnalysis()
    clf.fit(data_train, labels_train)
    labels_pred = clf.predict(data_test)
    if pattern == 0:
        accuracy = metrics.accuracy_score(labels_test, labels_pred) * 100
    elif pattern == 1:
        accuracy = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_lda_model.m')

    return accuracy

acc_s, acc_k, acc_l, accstd_s, accstd_k, accstd_l = evaluate_band_selection(sorted_index, pattern,data_path,label_path)