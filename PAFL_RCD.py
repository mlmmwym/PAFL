import copy
import numpy.linalg as nl
import numpy as np
import torch
from skmultiflow.drift_detection import DDM,EDDM,ADWIN
from skmultiflow.drift_detection.hddm_a import HDDM_A
from skmultiflow.drift_detection.hddm_w import HDDM_W

class PAAlgorithm:
    def __init__(self,shape):
        self.w = [];
        self.sum = 0;
        self.cansu=1.75
        self.id=1

        for i in range(shape+1):
            self.w.append(0)

    def updata_id(self,id):
        self.id=id

    def training_a_data(self,data, label):
        x = data
        y = label;
        margin = y * (np.dot(self.w, x));
        hinge_loss = max(0, 1 - margin);

        if (np.dot(x, x) != 0):
            tao = hinge_loss * self.cansu / (np.dot(x, x))
        else:
            tao = 0;

        self.w += tao * y * x;


    def prediction(self,data,label):
        x = data
        y = label;
        margin = y * (np.dot(self.w, x));
        if (margin > 0): return True
        return False

    def loss_is_0(self,x,y):
        if 1-y * (np.dot(self.w, x))>0:
            return False
        return True


    def get_forget_model(self,data,loss):

        for i in range(len(data)-1,-1,-1):
            x=data[i][:-1].reshape(-1,1)
            y=data[i][-1]
            if(loss[i]==1):
             self.w=get_new_w(x,y,np.array(self.w).reshape(-1,1),self.cansu)



def get_Vector_Mod(x):
    a = np.dot(x.reshape(-1), x.reshape(-1))
    return a.copy()

def get_new_w(x, y, w, cansu):
    global dd
    k = np.dot(y / get_Vector_Mod(x), x)
    k = np.dot(cansu, k)
    E = np.eye(x.shape[0])
    w = k - w
    A = np.dot(y, np.dot(k, x.reshape(1, -1))) - E
    return np.dot(nl.pinv(A), w).reshape(-1).copy()

class PAFL_RCD:
    def __init__(self,preq_data, preq_label):
        self.preq_data=preq_data
        self.preq_label=preq_label
        self.classifier_pool=[]
        self.classifier_pool_data=[]
        self.activation_classifier=None
        self.activation_pool_data=[]
        self.DDM=DDM()
        # self.DDM=HDDM_W()
        # self.DDM = HDDM_A()
        self.kuai=300
        self.correct_num=0
        self.real_time_accuracy=[]
        self.new_classifier=None
        self.new_pool_data = []

        self.loss_is_dayu_0=[]
        self.forget_data=[]


        self.tan=0.5

        self.sum=0

        self.ant=0

    def find_C(self,forget_d):
        C={}

        for j in range(len(self.classifier_pool)):
            ans=0
            for i in range(len(forget_d)):  #1
            # for i in range(len(self.classifier_pool_data[j])): #2

                x=forget_d[i][:-1] #1
                y=forget_d[i][-1]
                if self.classifier_pool[j].prediction(x,y):
                     ans=ans+1
                # x = self.classifier_pool_data[j][i][:-1] #2
                # y = self.classifier_pool_data[j][i][-1]
                # if self.new_classifier.prediction(x,y):
                #      ans=ans+1

            if(len(forget_d))!=0: #1
             C[j]=ans/len(forget_d)
            # if len(self.classifier_pool_data[j]) != 0:#2
            #     C[j] = ans / len(self.classifier_pool_data[j])

        C=sorted(C.items(),key=lambda x:x[1],reverse=True)
        if(len(C)!=0):
            optimal_C,acc=C[0]
            if acc>=self.tan:
                return optimal_C
            return -1
        else:
            return -1



    def  training(self):
        id=1

        self.classifier_pool.append(PAAlgorithm(self.preq_data.shape[1]))
        self.classifier_pool_data.append([])
        self.activation_classifier=self.classifier_pool[0]
        self.activation_pool_data=self.classifier_pool_data[0]

        for i in range(len(self.preq_data)):
            x = np.append(self.preq_data[i], 1);
            y = self.preq_label[i];

            yuce=self.activation_classifier.prediction(x,y)
            if yuce:
                self.correct_num += 1;
                self.DDM.add_element(0)
            else:
                self.DDM.add_element(1)

            if ((i + 1) % self.kuai == 0):
                self.real_time_accuracy.append(self.correct_num / self.kuai)
                self.sum=self.sum+self.correct_num / self.kuai
                self.correct_num = 0;

            if self.DDM.detected_warning_zone():
                 self.forget_data.append(np.append(x,y).astype(np.float32))
                 self.loss_is_dayu_0.append(self.activation_classifier. loss_is_0(x,y))
                 print(
                         'Warning zone has been detected in data: ' + ' - of index: ' + str(
                             i))

                 if self.new_classifier is None:
                     self.new_classifier=PAAlgorithm(self.preq_data.shape[1])
                     self.new_pool_data.append(np.append(x,y))
                 self.new_classifier.training_a_data(x,y)

            elif self.DDM.detected_change() or len(self.forget_data)>=100:
                 print('Change has been detected in data: ' + ' - of index: ' + str(i)+" "+str(len(self.forget_data)))
                 #forget
                 self.activation_classifier.get_forget_model(self.forget_data,self.loss_is_dayu_0)

                 print(i,len(self.forget_data))
                 id_C=self.find_C(self.forget_data)
                 if id_C!=-1:
                    self.activation_classifier=self.classifier_pool[id_C]
                    self.activation_pool_data=self.classifier_pool_data[id_C]
                    print('chongxian',self.activation_classifier.id,'abc',len(self.classifier_pool))

                 else:
                    id+=1
                    self.classifier_pool.append(copy.deepcopy(self.new_classifier))
                    self.classifier_pool_data.append(copy.deepcopy(self.new_pool_data))

                    self.activation_classifier = self.classifier_pool[-1]
                    self.activation_pool_data = self.classifier_pool_data[-1]

                    self.activation_classifier.updata_id(id)
                    print('new', self.activation_classifier.id, 'abc', len(self.classifier_pool))

                 self.forget_data = []
                 self.loss_is_dayu_0 = []
            else:
                if not self.new_classifier:
                    self.new_classifier=None
                    self.new_pool_data=[]
                self.forget_data=[]
                self.loss_is_dayu_0=[]
                self.activation_pool_data.append(np.append(x,y))
            self.activation_classifier.training_a_data(x,y)
        print(self.sum/(len(self.preq_data)/self.kuai))
        self.ant=self.sum/(len(self.preq_data)/self.kuai)









def get_synthetic_data(data, id):
    path = 'E:/科研/第二个工作/PAFL_RCD数据集/' + data + str(id) + '.npy'
    data = np.load(path)
    data = torch.from_numpy(data)
    data = data.float()
    preq_data = data[:, 0:-1]
    preq_label = data[:, -1]


    preq_label = preq_label.long()
    return preq_data, preq_label

import scipy
def get_synthetic_data1(data,id):
    path = 'E:/科研/第一个工作/第一个工作/数据集/'+data+'/' + data+'.mat'
    data = scipy.io.loadmat(path)
    data = data.get('data')
    data = torch.from_numpy(data)
    data=data.float()
    # preq_data = data[:, 0:-1]
    # preq_label = data[:, -1]

    kuai=(int)(len(data)/10)
    preq_data=None
    preq_label=None
    # preq_data = torch.cat([preq_data, preq_data1], 0)
    # preq_label = torch.cat([preq_label, preq_label1], 0)
    for i in range(10):
        if i==id:continue
        if preq_data is None:
            preq_data=data[i*kuai:(i+1)*kuai, 0:41]
            preq_label = data[i*kuai:(i+1)*kuai, 41:]
            a,preq_label=torch.max(preq_label,1)
        else:
            preq_data = copy.deepcopy(torch.cat([preq_data, data[i*kuai:(i+1)*kuai, 0:41]], 0))
            x=data[i*kuai:(i+1)*kuai, 41:]
            aaaaaa,x=torch.max(x,1)
            preq_label = copy.deepcopy(torch.cat([preq_label, x], 0))
    preq_label = preq_label.long()
    return preq_data, preq_label

if __name__ == '__main__':
    synthetic_data = ['RandomTree','Sine','Stagger','RandomRBF','Mixed','Sea','Stagger']#kddcup
    synthetic_data=['RandomRBF']
    for data in synthetic_data:
        epco=10
        accuracy=[]
        for ep in range(epco):
            print(ep,"*"*20)
            preq_data, preq_label = get_synthetic_data(data, ep)
            print(preq_data.shape)
            print(len(preq_data))
            preq_data = preq_data.numpy().astype(np.float64)
            preq_label = preq_label.numpy()
            preq_label = np.where(preq_label == 0, -1, preq_label)
            Model=PAFL_RCD(preq_data,preq_label)
            Model.training()
            if len(accuracy)==0:
                for i in range(len(Model.real_time_accuracy)):
                    accuracy.append(Model.real_time_accuracy[i])
            else:
                for i in range(len(Model.real_time_accuracy)):
                    accuracy[i]+=Model.real_time_accuracy[i]

            # np.save('E:/科研/第二个工作/PAFL_RCD多次实验结果/'+str(ep)+'_' + data + '_PA_RCD.npy', Model.real_time_accuracy)
        for i in range(len(accuracy)):
            accuracy[i]/=epco
        print(np.mean(accuracy))
        # np.save('E:/科研/第二个工作/PAFL_RCD实验结果/' + data + '_PA_RCD.npy', accuracy)

