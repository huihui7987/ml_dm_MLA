# coding=utf-8
from numpy import *
from itertools import chain

def load_data(file_name):
    data_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        fltline = list(chain(map(float, curline)))
        data_mat.append(fltline)
    return data_mat


def regleaf(data):
    '''平均数'''
    return mean(data[:, -1])


def regerr(data):
    '''方差'''
    return var(data[:, -1]) * shape(data)[0]


def split_data_set(data, feature, value):
    '''以样本中某一值分类，大于这个值的为一类，小于等于的为另一类'''
    mat0 = data[nonzero(data[:, feature] > value)[0], :]
    # nonzero返回非零元素行列中坐标
    # data[nonzero]可根据坐标返回矩阵中的非零元素
    mat1 = data[nonzero(data[:, feature] <= value)[0], :]
    return mat0, mat1


def choose_best_split(data, leaftype=regleaf, errtype=regerr,
                      ops=(1, 4)):  # ops是自定义值，用于控制函数停止的时机。ops[0]定义了总体方差与平均方差的最小值，ops[1]集合长度。
    '''选择最佳分类'''
    tols = ops[0]
    toln = ops[1]
    if len(set(data[:, -1].T.tolist()[0])) == 1:
        # 数据最后一列转成列表，并判断是否只有一个元素
        # 如果是就返回None和列表的平均数
        return None, leaftype(data)
    m, n = shape(data)
    s = errtype(data)
    # 数据最后一列的方差*行数=总体方差
    min_var = inf
    best_index = 0
    best_value = 0
    # min_var=正无穷，初始为正无穷是因为要让第一次判断无论如何都成立，这样才能继续下去
    for feat_index in range(n - 1):
        # range(n-1)=0,样本只有两列
        for split_value in set([float(i) for i in data[:, feat_index]]):
            # 分类值取自第一列数据的所组成的集合，集合具有互异性
            mat0, mat1 = split_data_set(data, feat_index, split_value)
            #
            if (shape(mat0)[0] < toln) or shape(mat1)[0] < toln: continue
            # 如果mat0的行数小于4或者mat1的行数小于4，则结束这次循环，下面代码不再执行
            #
            news = errtype(mat0) + errtype(mat1)
            # 俩子集方差相加
            if news < min_var:
                # 取最小方差，也就是波动最小，相似度较高
                best_index = feat_index
                best_value = split_value
                min_var = news
    if (s - min_var) < tols:
        # 总体方差-最小方差
        return None, leaftype(data)
    mat0, mat1 = split_data_set(data, best_index, best_value)
    if (shape(mat0)[0] < toln) or (shape(mat1)[0] < toln):
        # 如果mat0的行数小于4或者mat1的行数小于4
        return None, leaftype(data)
    return best_index, best_value


def create_tree(data, leaftype=regleaf, errtype=regerr, ops=(1, 4)):
    '''建立回归树'''
    feat, val = choose_best_split(data, leaftype, errtype, ops)
    # 分类特征，分类值
    if feat == None: return val
    # 如果分类特征为空，则返回分类值。应该理解为样本数据根据单一值就可以很好的划分。
    tree = {}
    tree['split_index'] = feat
    tree['split_value'] = val
    left_set, right_set = split_data_set(data, feat, val)
    # 根据最佳分类获得左右两个集合
    # 大于的在左，小的在右。。这个可以自己改
    tree['left'] = create_tree(left_set, leaftype, errtype, ops)
    tree['right'] = create_tree(right_set, leaftype, errtype, ops)
    return tree




def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/ 2.0

def prune(tree,testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = split_data_set(testData, tree['split_index'], tree['split_value'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = split_data_set(testData, tree['split_index'], tree['split_value'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print ("merging")
            return treeMean
        else: return tree
    else: return tree

def test():
    data = load_data('/Users/ghuihui/PycharmProject/ml_dm_MLA/Ch09/ex2test.txt')
    data = mat(data)
    t = create_tree(data)
    print (t)
    prune_tree = prune(t,data)
    print(prune_tree)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x = [float(i) for i in data[:, 0]]
    y = [float(i) for i in data[:, 1]]
    ax.scatter(x, y)
    plt.show()
test()