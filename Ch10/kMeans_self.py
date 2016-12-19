#-*-coding:utf-8 -*-

import numpy as np
from itertools import chain
import random

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        readlines = fr.readlines()
    for line in readlines:
        curLine = line.strip().split('\t')
        fltLine = list(chain(map(float,curLine)))
        dataMat.append(fltLine)
    return np.mat(dataMat)

def distEclud(vecA,vecB):

    return np.sqrt(np.sum(np.power(vecA-vecB,2)))
    #return np.sqrt(np.sum(np.power(oo1 - oo2,2)))



def randCent(dataSet,k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        #point1 = dataSet[:,j]
        rangJ = float(max(dataSet[:,j]) - min(dataSet[:,j]))
        centroids[:,j] = min(dataSet[:,j]) + rangJ * np.random.rand(k,1)#k行，随机数
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))#create mat to assign data points
                                      #to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)#簇质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#for each data point assign it to the closest centroid
            minDist = np.inf#正无穷
            minIndex = -1
            for j in range(k):#找最近质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])#欧几里得距离（每一个原始数据跟某一次确立的质心求距离）
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j#即最小距离的质心的索引
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        #print (centroids)
        for cent in range(k):#recalculate centroids，更新质心
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster
            centroids[cent,:] = np.mean(ptsInClust, axis=0) #assign centroid to mean
    return centroids, clusterAssment



def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):#2<4 if k=4
        lowestSSE = np.inf
        for i in range(len(centList)):
            #kk = clusterAssment[:,0]
            #ll = clusterAssment[:,0].A==i
            #jj = np.nonzero(clusterAssment[:,0].A==i)[0]

            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1])
            print ("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print ('the bestCentToSplit is: ',bestCentToSplit)
        print ('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return np.mat(centList), clusterAssment





def test():
    data = loadDataSet('/Users/ghuihui/PycharmProject/ml_dm_MLA/Ch10/testSet2.txt')
    centroids, clusterAssment = biKmeans(data,3)
    #print(clusterAssment)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [float(i) for i in data[:, 0]]
    y = [float(i) for i in data[:, 1]]
    x_c = [float(i) for i in centroids[:,0]]
    y_c = [float(i) for i in centroids[:,1]]
    ax.scatter(x, y,marker = '^')
    ax.scatter(x_c,y_c,marker='o',color = 'red')
    plt.show()

#test()

def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180)
    b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) * np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('/Users/ghuihui/PycharmProject/ml_dm_MLA/Ch10/places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    #print('Done')
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

clusterClubs()
