import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy import stats
import math


def readImage(fp):
    img = cv2.imread(fp)
    # return np.asarray(img[..., ::-1])
    return np.asarray(img)

def segment(img, number):
    # rs = np.random.randint(0, 100)
    kmeans = KMeans(n_clusters=number, random_state=0).fit(img)
    print(kmeans.cluster_centers_)
    return kmeans.cluster_centers_

def EM(img, number, mus):

    # mus = np.random.uniform(0, 255, [number, 3])
    pis = np.zeros((number,))
    for i in range(number):
        pis[i] = 1/number
    w = np.zeros((len(img), number))
    it = 0                                  # number of iteration
    ##### E Step #######
    while(it < 20):
        it += 1
        print("Iteration: ", it)
        print(mus)
        print(pis)
        for px in range(len(img)):
            denom = 0
            for k in range(number):
                distk = img[px] - mus[k]
                denom += np.exp(-0.5 * np.dot(distk, distk) * pis[k])
            wj = np.zeros((number, ))
            for k in range(number):
                distk = img[px] - mus[k]
                wj[k] = np.exp(-0.5 * np.dot(distk, distk)) * pis[k]
            w[px, :] = wj/denom
            # wj = np.zeros((number,))
            # for k in range(number):
            #     distk = img[px] - mus[k]
            #     wj[k] = (-0.5) * distk * distk
            # max = np.max(wj)
            # wj -= max
            # wj = np.exp(wj)
            # for k in range(number):
            #     wj[k] *= pis[k]
            # wj /= (np.sum(wj))
            # w[px, :] = wj
            ###### log version #########
            # w_nom = np.zeros((number, ))
            # wj = np.zeros((number, ))
            # for k in range(number):
            #     distk = img[px] - mus[k]
            #     w_nom[k] = -0.5 * np.dot(distk, distk) + np.log(pis[k])
            # w_denom = np.zeros((number, ))
            # for k in range(number):
            #     distk = img[px] - mus[k]
            #     w_denom[k] = -0.5 * np.dot(distk, distk)
            # max = np.max(w_denom)
            # w_den = 0
            # for k in range(number):
            #     w_den += np.exp(w_denom[k] - max) * pis[k]
            # w_den = max + np.log(w_den)
            # w[px, :] = np.exp(w_nom - w_den)

    #     ###### M Step #######
        newmus = np.zeros((number, 3))
        newpis = np.zeros((number, ))
        for j in range(number):
            nom = 0
            # total = 0
            total = np.sum(w[:, j])
            for px in range(len(img)):
                nom += img[px] * w[px, j]
                # total += w[px, j]
            newmus[j] = nom/total
            newpis[j] = total/len(img)
        # if(np.abs(np.sum(mus) - np.sum(newmus)) < 0.4):
        #     converge = True
        print("diff: ", np.abs(np.sum(mus) - np.sum(newmus)))
        mus = newmus
        pis = newpis
    print("==========Result=============")
    print(mus)
    print(pis)
    return mus, pis

def findCluster(value, mus):
    min = math.inf
    cluster = None
    for i, mu in enumerate(mus):
        # print(value, mu)
        dist = np.linalg.norm(value-mu)
        if dist < min:
            cluster = mu
            min = dist
    return cluster

def toImage(oriImg, mus):
    newImg = np.zeros(np.shape(oriImg))
    for i in range(len(oriImg)):
        for j in range(len(oriImg[0])):
            closest = findCluster(oriImg[i, j], mus)
            newImg[i, j] = closest/255
    cv2.imshow('image', newImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    k = 10
    # tree_img = readImage("tree.jpg")
    flower_img = readImage("flower.jpg")
    # fish_img = readImage("fish.jpg")
    # sunset_img = readImage("sunset.jpg")
    # tree_flatten = np.reshape(tree_img, (-1, 3))
    flower_flatten = np.reshape(flower_img, (-1, 3))
    # fish_flatten = np.reshape(fish_img, (-1, 3))
    # tree_flatten = np.reshape(tree_img, (-1, 3))
    initial_mus = segment(flower_flatten, k)
    mus, pis = EM(flower_flatten, k, initial_mus)
    # pis = np.zeros((k,))
    toImage(flower_img, mus)

