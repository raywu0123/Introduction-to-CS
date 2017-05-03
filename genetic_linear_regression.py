__author__ = 'ray'
import matplotlib.pyplot as plt
import numpy as np

def loss(x_train,y_train,parameters):
    S=0
    A=parameters[:,0]
    B=parameters[:,1]

    for ind,x in enumerate(x_train):
        S+=(y_train[ind]-(A*x+B))**2
    return S
def loss_2(x_train,y_train,parameters):
    S=0
    A=parameters[0]
    B=parameters[1]

    for ind,x in enumerate(x_train):
        S+=(y_train[ind]-(A*x+B))**2
    return S
def init():
    num_train_point=10
    randomness=0
    instances_num=10

    x_train = np.arange(0, num_train_point, 1);
    y_train =3*np.arange(0,num_train_point,1)+randomness*np.random.rand(num_train_point)

    ##let regression line be y=Ax+B

    paras=np.random.rand(instances_num,2)
    losses=np.array([loss(x_train,y_train,paras)]).T
    instances=np.concatenate((paras,losses),axis=1)
    instances.dtype=[('A',float),('B',float),('L',float)]
    return x_train,y_train,instances

def train(instances):
    def reproduce(insts):
        paras=[]
        for inst1 in insts:
            for inst2 in insts:
                A=inst1[0][0]
                B=inst2[0][1]
                paras.append([A,B])
        paras=np.array(paras)
        losses=np.array([loss(x_train,y_train,paras)]).T
        new_insts=np.concatenate((paras,losses),axis=1)
        new_insts.dtype=[('A',float),('B',float),('L',float)]
        return new_insts
    def mutate(insts):
        mutate_rate=0.1
        mutate_amp=0.1
        for i in range(int(len(insts)*mutate_rate)):
            idx=np.random.randint(0,len(insts))
            A=insts[idx][0][0]
            B=insts[idx][0][1]
            mutation_A=(np.random.random()-0.5)*mutate_amp
            mutation_B=(np.random.random()-0.5)*mutate_amp
            L=loss_2(x_train,y_train,(A+mutation_A,B+mutation_B))
            insts[idx][0]=(A+mutation_A,B+mutation_B,L)

        return insts
    def select(inst):
        inst=np.sort(inst,axis=0,order='L')
        selected_instances=inst[0:10,:]
        return selected_instances
    instances=reproduce(instances)
    instances=mutate(instances)
    instances=select(instances)
    return instances

plt.show()

epochs=100
x_train,y_train,instances=init()
plt.plot(x_train, y_train,'r.')

for i in range(epochs):
    instances=train(instances)

    A=instances[0][0][0]
    B=instances[0][0][1]
    print('epoch '+str(i+1)+'   loss=',instances[0][0][2],'A=',A,'B=',B)
    y=A*x_train+B
    if i%5==0:    plt.plot(x_train,y,color=(float(i)/float(epochs),0,1-float(i)/float(epochs)))
    #if i==epochs-1:     plt.plot(x_train,y,'y')
plt.show()


