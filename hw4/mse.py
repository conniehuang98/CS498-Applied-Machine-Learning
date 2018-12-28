from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix

import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import manifold


def unpikle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def seperate(dict):
    return dict[b'data'], dict[b'labels']


def pca(data):
    pca = PCA(n_components=20)
    components = pca.fit_transform(data)
    reconstruct = pca.inverse_transform(components)
    return reconstruct


def divide(raw_data, labels, data):
    for i in range(len(labels)):
        if labels[i] not in data.keys():
            data[labels[i]] = [raw_data[i]]
        else:
            data[labels[i]].append(raw_data[i])
    return data

def mse(mat1, mat2):
    return 1/5000 * np.sum((mat1 - mat2) ** 2)


if __name__ == '__main__':

    raw_data1, labels1 = seperate(unpikle("data_batch_1"))
    raw_data2, labels2 = seperate(unpikle("data_batch_2"))
    raw_data3, labels3 = seperate(unpikle("data_batch_3"))
    raw_data4, labels4 = seperate(unpikle("data_batch_4"))
    raw_data5, labels5 = seperate(unpikle("data_batch_5"))


    # print(raw_data1[0], labels1[0])
    data = dict()
    data = divide(raw_data1, labels1, data)
    data = divide(raw_data2, labels2, data)
    data = divide(raw_data3, labels3, data)
    data = divide(raw_data4, labels4, data)
    data = divide(raw_data5, labels5, data)

    error = [0 for i in range(10)]
    meanImages = np.zeros((10, 3072))

    for i in range(10):
        meanImages[i] = np.mean(data[i], axis = 0)
        recon = pca(data[i])
        for j in range(len(data[i])):
            error[i] += mse(data[i][j], recon[j])

    plt.figure(0)
    plt.bar(range(1, len(error)+1), error)
    plt.show()

    #7.7 b
    dis_mat = distance_matrix(meanImages, meanImages)
    plt.figure(1)
    mds = manifold.MDS(dissimilarity='precomputed')
    results = mds.fit(dis_mat)
    coords = results.embedding_
    plt.scatter(coords[:, 0], coords[:, 1])

    plt.show()
