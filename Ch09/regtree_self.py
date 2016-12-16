import numpy as np
from itertools import chain

def loadDataSet(filename):
    with open(filename) as f:
        lines = f.readlines()
    dataMat = []
    for line in lines:
        data = line.strip().split('\t')
        data = list(chain(map(float,data)))
        dataMat.append(data)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]#np.nonzero()🔙返回非零元素的索引，
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]#var是均方差函数（平均，计算差值，再平方），此处返回总的方差，再乘以个数

def linearSoler(dataSet):
    m,n = np.shape(dataSet)
    X = np.mat(np.ones((m,n)))
    Y = np.mat(np.ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]
    Y = dataSet[:,-1]

    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('jsadijasijd')
    ws = xTx.I * (X.T*Y)
    return ws,X,Y
def modelLeaf(dataSet):
    ws,X,Y = linearSoler(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSoler(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat,2))

def chooseBestSplit(dataSet,leafType = regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0]
    tolN=ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:#
        return None,leafType(dataSet)

    m,n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0

    for featIndex in range(n-1):
        for splitVal in set([float(i) for i in dataSet[:, featIndex]]):

            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
                # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue  # returns the best feature to split on
    # and the value used for that split

def createTree(dataSet, leafType=modelLeaf, errType=modelErr, ops=(1,10)):#assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)#choose the best split
    if feat == None: return val #if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    return (type(obj).__name__ == 'dict')


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def test():
    data = loadDataSet('/Users/ghuihui/PycharmProject/ml_dm_MLA/Ch09/exp2.txt')
    data = np.mat(data)
    t = createTree(data)
    print (t)
    #prune_tree = prune(t,data)

    #print(prune_tree)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [float(i) for i in data[:, 0]]
    y = [float(i) for i in data[:, 1]]
    ax.scatter(x, y)
    plt.show()


test()