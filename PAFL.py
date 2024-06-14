import operator
import numpy.linalg as nl
import numpy as np;
import torch
from scipy.spatial import distance
import numpy as np
global new_w
global old_w
global loss_last
global dd
import copy
import Matrix
from decimal import Decimal


class meanStd(object):
    def __init__(self):
        self.mean = 0.0
        self.mean_old = 0.0

        self.S = 0.0
        self.S_old = 0.0

        self.std = 0.001

        self.M_old = 0.0
        self.M = 0.0

        self.count = 0.0
        self.minMean = 100.0
        self.minStd = 100.0


    def calcMeanStd(self, data, cnt=1):#当前时间为t
        self.data = data

        self.mean_old = copy.deepcopy(self.mean)#1-(t-1)的平均值
        self.M_old = self.count * self.mean_old#1-(t-1)之间的和
        # self.M = self.M_old + data#1-(t)之间的和
        self.S_old = copy.deepcopy(self.S)#1-(t-1)的方差

        if self.count > 0:
            self.S = self.S_old + ((self.count * data - self.M_old) ** 2) / (self.count * (self.count + cnt))#1-(t)的方差

        self.count += cnt
        self.mean = self.mean_old + np.divide((data - self.mean_old), self.count)#1-(t)的平均值
        self.std = np.sqrt(self.S / self.count)#1-(t)的标准差

    def get_new_Mean_std(self,data):
        next_mean_old = copy.deepcopy(self.mean)  # 1-(t-1)的平均值
        next_M_old = self.count * next_mean_old  # 1-(t-1)之间的和
        next_S_old = copy.deepcopy(self.S)  # 1-(t-1)的方差
        next_count=copy.deepcopy(self.count)

        if next_count > 0:
         next_S = next_S_old + ((next_count * data - next_M_old) ** 2) / (
                        next_count * (next_count + 1))  # 1-(t)的方差
        else:
            next_S=0

        next_count =next_count+ 1
        next_mean = next_mean_old + np.divide((data - next_mean_old), next_count)  # 1-(t)的平均值
        next_std = np.sqrt(next_S / next_count)  # 1-(t)的标准差
        next_minMean=min(next_mean,copy.deepcopy(self.minMean))
        next_minStd=min(next_std,copy.deepcopy(self.minStd))

        return next_minMean,next_minStd,next_mean,next_std

    def resetMinMeanStd(self):
        self.minMean = copy.deepcopy(self.mean)
        self.minStd = copy.deepcopy(self.std)

    def calcMeanStdMin(self):
        if self.mean < self.minMean:
            self.minMean = copy.deepcopy(self.mean)
        if self.std < self.minStd:
            self.minStd = copy.deepcopy(self.std)

def isNeedData(loss, minMeanloss, minStdloss, meanloss, stdloss):
    growNode = True

    dynamicKsigmaGrow = 4*(1.3 * np.exp(-loss) + 0.7)
    growCondition1 = minMeanloss + dynamicKsigmaGrow * minStdloss
    growCondition2 = meanloss + stdloss
    # print(minMeanloss + minStdloss,growCondition2,growCondition1,dynamicKsigmaGrow)
    if growCondition2 > growCondition1:
        growNode = False
    return growCondition2,growCondition1
    # return growNode


class PAAlgorithm:
    def __init__(self,cisi_preq_data, cisi_preq_label,cansu=1.75):
        self.preq_data = None;
        self.preq_label =None;
        self.cisi_preq_data = cisi_preq_data;
        self.cisi_preq_label = cisi_preq_label;
        self.w = [];
        self.correct_num = 0;
        self.kuai = 300;
        self.sum = 0;
        self.a = []
        self.b=[]
        self.cansu=cansu
        self.c=None
        self.LOSS=meanStd()
        for i in range(self.cisi_preq_data.shape[1]+1):
            self.w.append(0)

    def training_a_data(self,preq_data, preq_label):
        x = np.append(preq_data, 1);
        y = preq_label;
        margin = y * (np.dot(self.w, x));
        hinge_loss = max(0, 1 - margin);

        if (np.dot(x, x) != 0):
            tao = hinge_loss * self.cansu / (np.dot(x, x))
        else:
            tao = 0;

        self.w += tao * y * x;
        Loss = hinge_loss * copy.deepcopy(self.cansu)
        self.LOSS.calcMeanStd(Loss)
        self.LOSS.calcMeanStdMin()
        # self.LOSS.resetMinMeanStd()

    def training_data(self,preq_data, preq_label):
        global loss_last
        self.w=np.array(self.w).astype(np.float64)
        self.c=[]
        self.preq_data = preq_data;
        self.preq_label = preq_label;
        self.correct_num = 0;
        for i in range(len(self.preq_data)):
            x = np.append(self.preq_data[i], 1);
            y = self.preq_label[i];
            margin = y * (np.dot(self.w, x));
            if (margin > 0): self.correct_num += 1;
            if ((i + 1) % self.kuai == 0):
                self.sum += self.correct_num / self.kuai;
                self.correct_num = 0;
            hinge_loss = max(0,1 - margin);

            if (np.dot(x, x) != 0):
                tao = hinge_loss * self.cansu/(np.dot(x, x))
            else:
                tao = 0;

            # if(i==3299):
            #     print("+"*30)
            #     print(self.w)
            #     print(x)
            #     print(y)
            self.w += tao * y * x;
            # if(i==3299):
            #  print(self.w)
            #  print("+" * 30)

            if (i >= 3000):
                self.c.append(self.w.copy())
                if(np.abs(tao-0)<=1e-6): self.b.append(0)
                else: self.b.append(1)
            else:
                if self.LOSS.count == 200:
                    self.LOSS.resetMinMeanStd()
                Loss=hinge_loss * copy.deepcopy(self.cansu)
                a,b,c,d=self.LOSS.get_new_Mean_std(Loss)
                self.LOSS.calcMeanStd(Loss)
                self.LOSS.calcMeanStdMin()
                # if self.LOSS.count ==2999:
                #     self.LOSS.resetMinMeanStd()


    def prediction(self):
        self.correct_num = 0
        for ii in range(len(self.cisi_preq_data)):
            x = np.append(self.cisi_preq_data[ii], 1);
            y = self.cisi_preq_label[ii];
            margin = y * (np.dot(self.w, x));
            if (margin > 0): self.correct_num += 1


    def forgot_data(self,data,label):
        self.c=[]
        global dd
        dd=0
        #print("sum_b:",sum(self.b))
        for i in range(len(data)-1,-1,-1):
            self.c.append(self.w.copy())
            x=np.append(data[i], 1).reshape(-1,1)
            if(i!=0):
                x1=np.append(data[i-1], 1).reshape(-1, 1)
                y1=label[i-1]
            else:
                x1=x
                y1=label[i]
            if(self.b[i]==1):
             self.w=get_new_w(x,label[i],np.array(self.w).reshape(-1,1),self.cansu,i,x1,y1)
        #print("dd:",dd)


    def select_data(self,data,label):
        new_data={}
        for i in range(len(data)):
            x = np.append(data[i], 1)
            y = label[i]
            margin = y * (np.dot(self.w, x))
            Loss=max(0,1 - margin)
            a,b,c,d=self.LOSS.get_new_Mean_std(Loss)
            # print(i+3000,":")
            # if isNeedData(Loss,a,b,c,d) and margin>0:#next_minMean,next_minStd,next_mean,next_std
            #     new_data[i]=np.abs(Loss-2/3)
            a,b=isNeedData(Loss,a,b,c,d)
            if( Loss!=0):
                X_L=Loss*self.cansu/(np.dot(x, x))*y*x
                new_data[i]=np.dot(X_L,X_L)
        new_data=sorted(new_data.items(), key=lambda e:e[1], reverse=False)
        data=[]
        for key, value in new_data:
            data.append(key)
        return data

def get_Vector_Mod(x):
    a=np.dot(x.reshape(-1),x.reshape(-1))
    return a.copy()

# 定义迭代函数
def jacobi_iteration(A, b, x0, max_iterations):
        x = x0.copy()
        for k in range(max_iterations):
            for i in range(len(b)):
                x[i] = (b[i] - np.dot(A[i], x) + A[i, i] * x[i]) / A[i, i]
        return x

def get_new_w1(x,y,w,cansu,ii):
    # 定义系数矩阵A和右侧向量b
    k = np.dot(y / get_Vector_Mod(x), x)
    k=np.dot(cansu,k)
    E = np.eye(x.shape[0])
    w= k - w
    b=w.copy()
    b=b.reshape(-1)
    A = np.dot(y, np.dot(k, x.reshape(1, -1))) - E
    # 初始解向量x0
    x0 = np.zeros_like(b)

    # 迭代次数
    max_iterations = 1000

    # 求解
    solution = jacobi_iteration(A, b, x0, max_iterations)
    return solution

def get_new_w(x,y,w,cansu,ii,x1,y1):
    global dd
    k=np.dot(y/get_Vector_Mod(x),x)
    k=np.dot(cansu,k)
    E=np.eye(x.shape[0])
    w=k-w
    A=np.dot(y,np.dot(k,x.reshape(1,-1)))-E
    rank_A=np.linalg.matrix_rank(A)
    A1=np.append(A,w,axis=1)
    # Ma=Matrix.Matr(A1)
    # jie=Ma.get_juZhenJie()
    # new_w=np.array([]).reshape(-1)
    # for i in range(-1000,1000):
    #     new_w1=jie[:,0]*i+jie[:,1]
    #     # print(new_w1, y1*np.dot(x1.reshape(-1),new_w1.reshape(-1)))
    #     if (y*np.dot(x.reshape(-1),new_w1.reshape(-1)))<=1 and y1*np.dot(x1.reshape(-1),new_w1.reshape(-1))>0:
    #          new_w=new_w1.reshape(-1)
    #          break;
    #
    # if(len(new_w)==0):
    #     # print("空空空")
    #     return np.dot(nl.pinv(A), w).reshape(-1).copy()
    # else:
    #     # print(new_w,ii)
    #     return new_w
    # rank_A1=np.linalg.matrix_rank(A1)
    # print(rank_A,rank_A1,ii)
   #  return np.dot(nl.inv(A),w).reshape(-1)
    return np.dot(nl.pinv(A),w).reshape(-1).copy()


#[ 2.65443734e-01  3.11191202e-01  6.00076715e-04 -2.64665136e+00] [1.01996470e+21 1.89348802e+21 1.72313487e+21 4.30699811e+20]
def get_synthetic_data(data, id):
    num=300
    path = 'E:/科研/第二个工作/cesi数据集300/' + data + str(id) + '.npy'
    path2 = 'E:/科研/第二个工作/cesi数据集'+str(num)+'/' + data + str(id) + '.npy'
    path1 = 'E:/科研/第二个工作/cesi数据集300/cisi_' + data + str(id) + '.npy'
    data = np.load(path)
    data1 = np.load(path1)
    data2 = np.load(path2)
    data = torch.from_numpy(data)
    data1 = torch.from_numpy(data1)
    data2 = torch.from_numpy(data2)
    data = data.float()
    data1 = data1.float()
    data2 = data2.float()
    preq_data = data[:3000, 0:-1]
    preq_label = data[:3000, -1]

    preq_data1 = data2[3000:, 0:-1]
    preq_label1 = data2[3000:, -1]

    # preq_data1 = data2[0:num, 0:-1]
    # preq_label1 = data2[0:num, -1]

    preq_data=torch.cat([preq_data,preq_data1],0)
    preq_label = torch.cat([preq_label, preq_label1], 0)

    cisi_preq_data = data1[:, 0:-1]
    cisi_preq_label = data1[:, -1]

    preq_label = preq_label.long()
    cisi_preq_label = cisi_preq_label.long()
    return preq_data, preq_label, cisi_preq_data, cisi_preq_label


if __name__ == '__main__':
    synthetic_data = ['sea1',
                      'sine','sine1',
                      'RandomTree','RandomTree1','RandomTree2','RandomTree3','RandomTree4',
                      'Stagger','Stagger1','Stagger2','Stagger3','Stagger4',
                      'RandomRBF', 'RandomRBF1', 'RandomRBF2', 'RandomRBF3', 'RandomRBF4'
                      ,'MIXED' ]
    # synthetic_data=['sea1']
    for xxx in range(1):
        zzk=1.75
        for data1 in synthetic_data:
            # print(data1,":")
            epco = 10
            a = []
            b=0
            x=0
            y=0
            z=0
            o=0
            for ii in range(epco):
                preq_data, preq_label, cisi_preq_data, cisi_preq_label = get_synthetic_data(data1, ii)
               # print(preq_data.shape)
                preq_data = preq_data.numpy().astype(np.float64)
                preq_label = preq_label.numpy()
                preq_label = np.where(preq_label == 0, -1, preq_label)

                cisi_preq_data = cisi_preq_data.numpy().astype(np.float64)
                cisi_preq_label = cisi_preq_label.numpy()
                # print(np.sum(cisi_preq_label),len(cisi_preq_label))
                cisi_preq_label=np.where(cisi_preq_label == 0, -1,cisi_preq_label)

                PA=PAAlgorithm( cisi_preq_data, cisi_preq_label,zzk)
                PA.training_data(preq_data, preq_label)
                PA.prediction()
                xx=PA.correct_num/len(cisi_preq_data)
                zhou=PA.c.copy()

                new_preq_data = preq_data[:3000].copy()
                new_preq_label = preq_label[:3000].copy()
                PA1 = PAAlgorithm( cisi_preq_data, cisi_preq_label,zzk)
                PA1.training_data(new_preq_data, new_preq_label)
                PA1.prediction()
                yy=PA1.correct_num/len(cisi_preq_data)

                PA.forgot_data(preq_data[3000:].copy(),preq_label[3000:].copy())
                PA.prediction()
                zz= PA.correct_num / len(cisi_preq_data)

                #第二步
                # zhou1=PA.c.copy()
                #
                # new_concept_data=preq_data[3000:].copy()
                # new_concept_label = preq_label[3000:].copy()
                # need_data=PA.select_data(new_concept_data,new_concept_label)

                x+=xx
                y+=yy
                z+=zz
                # np.save('E:/科研/第二个工作/PAFL_1000实验结果/1'+str(ii) + data1 + '_PASF1.npy',zz)
                # np.save('E:/科研/第二个工作/PAFL_1000实验结果/0'+str(ii) + data1 + '_PASF1.npy', xx)
            x/=epco
            y/=epco
            z/=epco
            print(data1,"\n","PAC_1_2:",x,"PAC_1:",y,"PAFL:",z)

#1是遗忘，0是没有遗忘
#PAFL1参数是1.75，PASF2参数是1





