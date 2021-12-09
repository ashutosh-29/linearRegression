from copy import deepcopy
from random import randint
import numpy as np
from matplotlib import pyplot
from sklearn.datasets.samples_generator import make_blobs
def dist(a, b):
    return np.linalg.norm(a - b)
def dist_2d(a,b):
    squared_distance = 0
    for i in range(len(a)):
        squared_distance += (a[i] - b[i])**2
    dist = sqrt(squared_distances)
    return dist

fake_centers = 4

x, y = make_blobs(n_samples=5000, centers=fake_centers, n_features=2, random_state=195)

x2 = deepcopy(x)
y2 = deepcopy(y)
# plot regression dataset
%matplotlib inline
pyplot.scatter(x[:, 0], x[:, 1])
pyplot.show()
number_of_clusters = fake_centers
def generate_random_clusters(n_features, k):
    c_position = []
    for i in range(n_features):
        c_position.append(np.random.randint(0.8 * np.min(x[:, i]), 0.8 * np.max(x[:, i]), size=k)) 
                                           # 0.8 to stay in the range (it really doesn't matter as this
                                           #                           is random initialization)
    return c_position
n_features = 2
c_positions = generate_random_clusters(n_features, number_of_clusters) 
c_positions = np.array(c_positions).T
print(c_positions)
print(x)
pyplot.scatter(x[:, 0], x[:, 1])
pyplot.scatter(c_positions[:, 0], c_positions[:, 1], marker='*', s=300, c='orange')
pyplot.show()
def error(c, c_old):
    return dist(c, c_old)
def kmeans(x, c_positions, number_of_clusters):
    clusters = np.zeros(len(x))
    old_clusters = np.zeros(c_positions.shape)
    error = 1
    while error != 0:
        for i in range(len(x)):
            distances = [dist(x[i], c) for c in c_positions]
            cluster = np.argmin(distances)
            clusters[i] = cluster        
        
        old_clusters = deepcopy(c_positions)
    
        for j in range(0, number_of_clusters):
            points = [x[n] for n in range(len(x)) if clusters[n] == j]
            c_positions[j] = np.mean(points, axis=0)        
        
        error = dist(c_positions, old_clusters)
        
    return c_positions, clusters
new_clusters, p = kmeans(x, c_positions, number_of_clusters)
print(new_clusters)
pyplot.scatter(x[:, 0], x[:, 1])
pyplot.scatter(new_clusters[:, 0], new_clusters[:, 1], marker='*', s=300, c='r')
pyplot.show()
def plot_in_col(x, number_of_clusters, p, new_clusters):
    for i in range(number_of_clusters):
        col_points = np.array([x[n] for n in range(len(x)) if p[n] == i])
        pyplot.scatter(col_points[:, 0], col_points[:, 1], s=10)
    pyplot.scatter(new_clusters[:, 0], new_clusters[:, 1], marker='*', s=300, c='r')
    pyplot.show()

plot_in_col(x, number_of_clusters, p, new_clusters)