import math
import numpy as np
import scipy.io as sio
def classify(test_data,train_data,train_target,k):
    dist=[]
    m,n=train_data.shape
    # 计算测试数据到每个点的欧式距离
    for i in range(m):
        temp=test_data-train_data[i]
        sum_2=0
        for j in range(len(temp)):
            sum_2+=temp[j]**2
        each_dist=math.sqrt(sum_2)
        dist.append(each_dist) #测试样本到index=i训练样本的距离被存放在res列表的index=i的位置，index=i的样本是第i+1个样本
    sorted_dist=sorted(dist)

    # k个最近的值所属的类别
    Class_count={}
    for i in range(k):
        label=train_target[dist.index(sorted_dist[i])]
        if label  in Class_count.keys():
            Class_count[label]+=1
        else:
            Class_count[label]=1
    Sorted_Class_count=sorted(Class_count.items(), key=lambda x: x[1], reverse=True)
    return Sorted_Class_count[0]

def creatDataSet():
    group=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
# 数据归一化
def autoNorm(dataset):
    m,n=dataset.shape # 注意.shape()与.shape的区别
    min_value=dataset.min(0)
    ''' .min()无参，返回所有中的最小值；.min(0)返回每列的最小值；.min(1)返回每行的最小值'''
    max_value=dataset.max(0)
    ranges=max_value-min_value
    normDataset=np.zeros((m,n))
    normDataset=dataset-min_value
    normDataset=normDataset/ranges
    return normDataset,ranges,min_value

# 定义测试算法的函数。
"""
衡量算法的准确性knn算法可以用正确率或者错误率来衡量。错误率为0，表示分类良好。因此可以将训练样本中的
10%用于测试，90%用于训练。
"""
#def datingClassTest():
hoRatio=0.1
matfn = 'data/balance_scale.mat' #数据集的路径
data = sio.loadmat(matfn) #将.mat数据集转换为适用于python的数据集（应该是.npy）
dataset = data['data'] #通过设置断点调试，查看data数据，发现里面的data属性是需要的数据集
Datasets = dataset[:, 0:-1] # 只含有特征的样本集
Labels = dataset[:, -1] # 样本集的标签集
m,n=Datasets.shape #注意.shape与.shape()的区别
test_data_num=int(m*hoRatio)
normDatasets,ranges,min_value=autoNorm(Datasets)
test_data=normDatasets[0,:]
test_target=Labels[0]
for i in range(1,5):
    test_data=np.vstack([test_data,normDatasets[i,:]])
    test_target=np.hstack([test_target,Labels[i]])
for i in range(2):
    for j in range(5):
        t=50*(i+1)+j
        test_data=np.vstack([test_data,normDatasets[t,:]])
        test_target=np.hstack([test_target,Labels[t]])
train_data=np.delete(normDatasets,[0,1,2,3,4,50,51,52,53,54,100,101,102,103,104],axis=0)
# axis=0表示按行删除元素
train_target=np.delete(Labels,[0,1,2,3,4,50,51,52,53,54,100,101,102,103,104],axis=0)
errorCount=0
for i in range(len(test_data)):
    pred_label=classify(test_data[i,:],train_data,train_target,3)
    if (pred_label - test_target[i]).all():
        errorCount+=1
print('the total error rate is :',errorCount/float(test_data_num))

"""
如果不打乱原本的数据集，则测试结果为0.0，即错误率为零，因为在这个数据集，属于某一类别的样本全部集中在一起了，所以误差为零
"""
"""
if __name__=='__main__':
    datasets,labels=creatDataSet()
    pred_label=classify([0,0.2],datasets,labels,3)
    print(pred_label)
"""
"""
# 加载.mat数据集，并提取特征集Datasets和标签集Labels
matfn='data/iris.mat'
data=sio.loadmat(matfn)
dataset=data['data']
Datasets=dataset[:,0:-1] #从完整数据集中提取全部的样本特征部分
Labels=dataset[:,-1]#从完整数据集中提取全面的标签集
"""



