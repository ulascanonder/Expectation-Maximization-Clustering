import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
import scipy.stats as stats
import math

data = np.genfromtxt("data_set.csv", delimiter = ",")  
centroids = np.genfromtxt("initial_centroids.csv", delimiter = ",")
K = 5 #K determines the number of clusters formed. We'll for 5 clusters for this data. 
N = len(data[:,0])

def initiate_memberships(centroids, x):
    distances = spa.distance_matrix(x, centroids, p=2)
    memberships = np.argmin(distances, axis = 1)
    return memberships

def update_centroids(memberships, X):
    centroids = np.vstack([np.mean(X[memberships == k, :], axis = 0) for k in range(K)])
    return(centroids)

#I use that function to find the nearest point for the data points to calculate their prior probablities
def find_nearest(array,value): 
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

memberships = initiate_memberships(centroids, data)

m = np.zeros([K,2])
cov = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
for i in range(K):
    cluster = data[memberships == i, :]
    m[i] = [np.mean(cluster[:,0]), np.mean(cluster[:,1])]
    cov[i] = np.cov(cluster.T)

x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

D1 = stats.multivariate_normal.pdf(X_grid, mean = m[0], cov = cov[0])
D1 = D1.reshape((len(x1_interval), len(x2_interval)))
D1_original = stats.multivariate_normal.pdf(X_grid, mean = [0, 5], cov = [[4.8, 0], [0, 0.4]])
D1_original = D1_original.reshape((len(x1_interval), len(x2_interval)))
D2 = stats.multivariate_normal.pdf(X_grid, mean = m[1], cov = cov[1])
D2 = D2.reshape((len(x1_interval), len(x2_interval)))
D2_original = stats.multivariate_normal.pdf(X_grid, mean = [-5.5, 0], cov = [[0.4, 0], [0, 2.8]])
D2_original = D2_original.reshape((len(x1_interval), len(x2_interval)))
D3 = stats.multivariate_normal.pdf(X_grid, mean = m[2], cov = cov[2])
D3 = D3.reshape((len(x1_interval), len(x2_interval)))
D3_original = stats.multivariate_normal.pdf(X_grid, mean = [0, 0], cov = [[2.4, 0], [0, 2.4]])
D3_original = D3_original.reshape((len(x1_interval), len(x2_interval)))
D4 = stats.multivariate_normal.pdf(X_grid, mean = m[3], cov = cov[3])
D4 = D4.reshape((len(x1_interval), len(x2_interval)))
D4_original = stats.multivariate_normal.pdf(X_grid, mean = [5.5, 0], cov = [[0.4, 0], [0, 2.8]])
D4_original = D4_original.reshape((len(x1_interval), len(x2_interval)))
D5 = stats.multivariate_normal.pdf(X_grid, mean = m[4], cov = cov[4])
D5 = D5.reshape((len(x1_interval), len(x2_interval)))
D5_original = stats.multivariate_normal.pdf(X_grid, mean = [0, -5.5], cov = [[4.8, 0], [0, 0.4]])
D5_original = D5_original.reshape((len(x1_interval), len(x2_interval)))



def update_memberships(centroids, X):
    memberships = np.zeros([N])
    for i in range(N):
        nearest1 = find_nearest(x1_interval, data[i][0])
        nearest2 = find_nearest(x2_interval, data[i][1])
        index1 = np.where(x1_interval == nearest1)
        index2 = np.where(x2_interval == nearest2)
        prior1 = D1[index1, index2] 
        prior2 = D2[index1, index2] 
        prior3 = D3[index1, index2] 
        prior4 = D4[index1, index2] 
        prior5 = D5[index1, index2] 
        memberships[i] = np.argmax([prior1, prior2, prior3, prior4, prior5])
        
    return memberships

def update_pdfs(memberships, X):
    m = np.zeros([K,2])
    cov = [[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]],[[0,0],[0,0]]]
    for i in range(K):
        cluster = X[memberships == i, :]
        m[i] = [np.mean(cluster[:,0]), np.mean(cluster[:,1])]
        cov[i] = np.cov(cluster.T)


    D1 = stats.multivariate_normal.pdf(X_grid, mean = m[0], cov = cov[0])
    D1 = D1.reshape((len(x1_interval), len(x2_interval)))
    D2 = stats.multivariate_normal.pdf(X_grid, mean = m[1], cov = cov[1])
    D2 = D2.reshape((len(x1_interval), len(x2_interval)))
    D3 = stats.multivariate_normal.pdf(X_grid, mean = m[2], cov = cov[2])
    D3 = D3.reshape((len(x1_interval), len(x2_interval)))
    D4 = stats.multivariate_normal.pdf(X_grid, mean = m[3], cov = cov[3])
    D4 = D4.reshape((len(x1_interval), len(x2_interval)))
    D5 = stats.multivariate_normal.pdf(X_grid, mean = m[4], cov = cov[4])
    D5 = D5.reshape((len(x1_interval), len(x2_interval)))
    
    return [D1, D2, D3, D4, D5]

centroids = update_centroids(memberships, data)
[D1, D2, D3, D4, D5] = update_pdfs(memberships, data)


#%% Iterations

for i in range(100):
    memberships = update_memberships(centroids, data)
    centroids = update_centroids(memberships, data)
    [D1, D2, D3, D4, D5] = update_pdfs(memberships, data)


#%% Visualisation and means

cluster1 = data[memberships == 0]
cluster2 = data[memberships == 1]
cluster3 = data[memberships == 2]
cluster4 = data[memberships == 3]
cluster5 = data[memberships == 4]

print('Means: \n', centroids)


plt.figure(figsize = (8, 8))
plt.plot(cluster1[:,0],cluster1[:,1], ".", color="blue")
plt.plot(cluster2[:,0],cluster2[:,1], ".", color="green")
plt.plot(cluster3[:,0],cluster3[:,1], ".", color="red")
plt.plot(cluster4[:,0],cluster4[:,1], ".", color="orange")
plt.plot(cluster5[:,0],cluster5[:,1], ".", color="purple")

plt.plot(centroids[:,0],centroids[:,1], ".", markersize = 12, color="red")
plt.contour(x1_grid, x2_grid, D1, levels = [0.05],
            colors = "blue")
plt.contour(x1_grid, x2_grid, D2, levels = [0.05],
            colors = "green")
plt.contour(x1_grid, x2_grid, D3, levels = [0.05],
            colors = "red")
plt.contour(x1_grid, x2_grid, D4, levels = [0.05],
            colors = "orange")
plt.contour(x1_grid, x2_grid, D5, levels = [0.05],
            colors = "purple")
plt.contour(x1_grid, x2_grid, D1_original, levels = [0.05],
            colors = "k", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D2_original, levels = [0.05],
            colors = "k", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D3_original, levels = [0.05],
            colors = "k", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D4_original, levels = [0.05],
            colors = "k", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D5_original, levels = [0.05],
            colors = "k", linestyles = "dashed")





