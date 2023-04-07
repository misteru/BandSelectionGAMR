import numpy as np
from sklearn.metrics.pairwise import *

def generate_graphs(X, n_neighbors):

    K = 9   #相似度矩阵个数
    (n, b) = X.shape
    #print('d...')
    n_neighbors = int(n_neighbors)
    Si = np.zeros([n,n,K])
    
    Si[:, :, 0] = laplacian_kernel(X)
#     Si[:, :, 0] = euclidean_distances(X)
    Si[:, :, 1] = cosine_similarity(X)
    Si[:, :, 1][Si[:, :, 1] < 0] = 0
    Si[:, :, 2] = rbf_kernel(X=X)
#     Si[:, :, 3] = rbf_kernel(X=X, Y=None, gamma=0.5)
#     Si[:, :, 4] = rbf_kernel(X=X, Y=None, gamma=5)
#     Si[Si < 0] = 0

    Si[:, :, 3] = laplacian_kernel(X, gamma=0.0001)
    Si[:, :, 4] = Si[:, :, 1]
    Si[:, :, 5] = rbf_kernel(X=X, gamma=0.0001)

    Si[:, :, 6] = laplacian_kernel(X, gamma=100)
    Si[:, :, 7] = Si[:, :, 1]
    Si[:, :, 8] = rbf_kernel(X=X, gamma=100)

    #构建KNN图
    n_neighbors = 20
    for k in range(3,6):
        index_sorted = np.argsort(Si[:, :, k])
        for i in range(n):
            for j in index_sorted[i][:-n_neighbors]:
                Si[i, j, k] = 0
                
    n_neighbors = 60
    for k in range(6,9):
        index_sorted = np.argsort(Si[:, :, k])
        for i in range(n):
            for j in index_sorted[i][:-n_neighbors]:
                Si[i, j, k] = 0

#     index_sorted = np.argsort(Si[:, :, 0])
#     for i in range(n):
#         for j in index_sorted[i][2:]:
#             Si[i, j, 0] = 0


    #行归一化
    #for k in range(K):
     #   D = np.diag(np.sum(Si[:, :, k], axis=1))
      #  Si[:, :, k] = np.linalg.inv(D) @ Si[:, :, k]
    
    return Si

Si = generate_graphs(a, b)
