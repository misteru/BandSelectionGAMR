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
    acc = np.zeros(12)
    
    iteration = 1
    for i in range(iteration):
        data_mat_new = data_mat[:,sorted_index.astype(int)]
        num = 10
        a = np.zeros(num)
        b = np.zeros(num)
        c = np.zeros(num)
        a1 = np.zeros(num)
        b1 = np.zeros(num)
        c1 = np.zeros(num)
        for i in range(num):
            a[i],a1[i] = eval_band_SVM(data_mat_new, label_mat, pattern)
            b[i],b1[i] = eval_band_KNN(data_mat_new, label_mat, pattern)
            c[i],c1[i] = eval_band_LDA(data_mat_new, label_mat, pattern)
        acc[0] +=  np.mean(a)
        acc[1] += np.mean(b)
        acc[2] += np.mean(c)
        acc[3] +=  np.std(a)
        acc[4] += np.std(b)
        acc[5] +=  np.std(c)
        
        acc[6] +=  np.mean(a1)
        acc[7] += np.mean(b1)
        acc[8] += np.mean(c1)
        acc[9] +=  np.std(a1)
        acc[10] += np.std(b1)
        acc[11] +=  np.std(c1)
        
    acc = np.ndarray.round(acc/iteration, 2)
    
    return acc
    
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
    accuracy1 = metrics.accuracy_score(labels_test, labels_pred) * 100
    accuracy2 = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_lda_model.m')

    return accuracy1,accuracy2

def eval_band_KNN(data_mat, labels_mat, pattern):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
    neigh.fit(data_train, labels_train)
    labels_pred = neigh.predict(data_test)
    accuracy1 = metrics.accuracy_score(labels_test, labels_pred) * 100
    accuracy2 = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_lda_model.m')

    return accuracy1,accuracy2

def eval_band_LDA(data_mat, labels_mat, pattern):

    data_train, data_test, labels_train, labels_test = train_test_split(data_mat, labels_mat, test_size=0.9, stratify = labels_mat)

    #模型训练与拟合
    clf = LinearDiscriminantAnalysis()
    clf.fit(data_train, labels_train)
    labels_pred = clf.predict(data_test)
    accuracy1 = metrics.accuracy_score(labels_test, labels_pred) * 100
    accuracy2 = metrics.cohen_kappa_score(labels_test, labels_pred) * 100 
    #存储模型
    #joblib.dump(neigh, 'work/model/Indian_pines_lda_model.m')

    return accuracy1,accuracy2

acc = evaluate_band_selection(sorted_index, pattern,data_path,label_path)