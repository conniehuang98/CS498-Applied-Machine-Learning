from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix

import numpy as np
import pickle
from matplotlib import pyplot as plt
from sklearn import manifold

def unpikle(file):
  with open(file, 'rb') as fo:
  dict = pickle.load(fo, encoding = 'bytes')
return dict

def pca(data):
  pca = PCA(n_components=20)
