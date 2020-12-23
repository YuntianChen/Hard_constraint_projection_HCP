import torch
from torch.nn import Linear,Tanh,Sequential,ReLU,Softplus
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as func
import random
from pyDOE import lhs
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from scipy.interpolate import griddata
import re
import os
import os.path
import operator
import sys
import dataset
import configure as config
import HCProjection as HCP
import simulation as simu

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

device = config.device
BATCH_SIZE_hyper = config.BATCH_SIZE_hyper
use_HCP = config.use_HCP
use_ANN = config.use_ANN
num_epoch = config.num_epoch_hyper
Nf_hyper = config.Nf_hyper
N_boun_hyper = config.N_boun_hyper
nx = config.nx     #网格个数
ny = config.ny
nt = config.nt      #时间步数

k = simu.k
x = simu.x
y = simu.y
t = simu.t
logk = simu.logk
h_boun1_0 = simu.h_boun1_0-200
h_boun2_0 = simu.h_boun2_0-200
hh = simu.hh-200
Ss = simu.Ss

H_boun_train = dataset.H_boun_train  
H_boun2_train = dataset.H_boun2_train
H_ic_train = dataset.H_ic_train
H_ic2_train = dataset.H_ic2_train
Y = dataset.Y
X = dataset.X
T = dataset.T
TXY_boun_train = dataset.TXY_boun_train
TXY_boun2_train = dataset.TXY_boun2_train
TXY_noflow1 = dataset.TXY_noflow1
TXY_noflow1_ghost = dataset.TXY_noflow1_ghost
TXY_noflow2 = dataset.TXY_noflow2
TXY_noflow2_ghost = dataset.TXY_noflow2_ghost
TXY_noflow = dataset.TXY_noflow
TXY_noflow_ghost = dataset.TXY_noflow_ghost
TXY_ic_train = dataset.TXY_ic_train
TXY_ic2_train = dataset.TXY_ic2_train
n_train0 = dataset.n_train0
n_colloc = dataset.n_colloc
TXY_f_train1 = dataset.TXY_f_train1
TXY_f_train2 = dataset.TXY_f_train2 # 去除边界后的配点，对应所有时间步
TXY_f_train3 = dataset.TXY_f_train3
TXY_f_train4 = dataset.TXY_f_train4
TXY_f_train5 = dataset.TXY_f_train5
TXY_f_train6 = dataset.TXY_f_train6
TXYK_kxky_f_train = dataset.TXYK_kxky_f_train
TXY_train = dataset.TXY_train
H_train = dataset.H_train

#########################################################
#建立模型
#########################################################
#设置保留小数位数
torch.set_printoptions(precision=7, threshold=None, edgeitems=None, linewidth=None, profile=None)
#定义神经网络
Net_pinn=Sequential(
    Linear(3,50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50,50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50,50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50, 50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50, 50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50, 50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50, 50),
    #torch.nn.BatchNorm1d(30, momentum=0.5),
    Softplus(beta=1, threshold=20),
    Linear(50, 1),
)
#网络参数初始化
def init_weights(m):
    if type(m) == Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
#初始化网络
Net_pinn.apply(init_weights)
#定义Loss
loss_pinn_set=[]
f1_set=[]
f2_set=[] # AH
f2updated_set=[] # 映射后的AH_ 
h_diff_set=[] # (H-H_)/H
diff0_set = [] # a0-pde0
diff1_set = [] # a1-pde1
diff2_set = [] # a2-pde2
pde012_set = []
a012_set = []
f2_pde_set=[]
f3_set=[]
f4_set=[]  
f5_set=[]
f6_set=[]
f7_set=[]
f7_hc_set=[]
hc_loss_set=[]
hc_loss_project_set=[]
hc_loss_std_set=[]
hc_loss_project_std_set=[]
gradient_diff_set=[]
start_time = time.time()  
Net_pinn=Net_pinn.to(device)
# 定义神经网络优化器
optimizer2=torch.optim.Adam([
    {'params': Net_pinn.parameters()},
])

#########################################################
#训练模型
#########################################################
#训练数据加载与分批
BATCH_SIZE = BATCH_SIZE_hyper     # 批训练的数据个数
torch_dataset = Data.TensorDataset(TXY_train, H_train) # 前t_train个有观测值的时间步的数据
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 要不要打乱数据 (打乱比较好)
    num_workers=0,              # 多线程来读数据
)
N_batch=math.ceil(n_train0/BATCH_SIZE) # 观测值数目/batch size获得批数目
BATCH_SIZE2=math.ceil(n_colloc/N_batch ) # 根据批数目计算每批中的配点数目
TXY_f_train_set1=[]
TXY_f_train_set2=[]
TXY_f_train_set3=[]
TXY_f_train_set4=[]
TXY_f_train_set5=[]
TXY_f_train_set6=[]
if use_HCP:
    for i_batch in range(int(N_batch)):
        TXY_f_train_set1.append(TXY_f_train1[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
        TXY_f_train_set2.append(TXY_f_train2[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
        TXY_f_train_set3.append(TXY_f_train3[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
        TXY_f_train_set4.append(TXY_f_train4[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
        TXY_f_train_set5.append(TXY_f_train5[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
        TXY_f_train_set6.append(TXY_f_train6[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])

TXYK_kxky_f_train_set=[]
for i_batch in range(int(N_batch)):
    TXYK_kxky_f_train_set.append(TXYK_kxky_f_train[BATCH_SIZE2*i_batch:BATCH_SIZE2*(i_batch+1),:])
start_time = time.time()  
#迭代训练
for epoch in range(num_epoch):   # 训练所有!整套!数据
    for ite, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        batch_x= Variable(batch_x, requires_grad=True).to(device) # 抽取的有观测的数据，包含边界
        batch_y= Variable(batch_y).to(device) # 抽取的有观测的数据，包含边界
        if use_HCP == True:
            batch_xf1=Variable(TXY_f_train_set1[ite],requires_grad=True).to(device) # 配点数据，全部时间步，无边界       
            batch_xf2=Variable(TXY_f_train_set2[ite],requires_grad=True).to(device) # 配点数据，全部时间步，无边界
            batch_xf3=Variable(TXY_f_train_set3[ite],requires_grad=True).to(device) # 配点数据，全部时间步，无边界
            batch_xf4=Variable(TXY_f_train_set4[ite],requires_grad=True).to(device) # 配点数据，全部时间步，无边界
            batch_xf5=Variable(TXY_f_train_set5[ite],requires_grad=True).to(device) # 配点数据，全部时间步，无边界
            batch_xf6=Variable(TXY_f_train_set6[ite],requires_grad=True).to(device) # 配点数据，全部时间步，无边界
        batch_xf=Variable(TXYK_kxky_f_train_set[ite],requires_grad=True).to(device)
        batch_x2= batch_xf[:,0:3]       
        K=batch_xf[:,-3:-2]
        k_x_=batch_xf[:,-2:-1]
        k_y_=batch_xf[:,-1:] 
        if use_HCP == True:      
            n_train=len(batch_xf1)     
        else:
            n_train=len(batch_xf)   
        optimizer2.zero_grad()
        # 预测观测

        
        def HCP_layer_observation (T_test_col, X_test_col, Y_test_col, k):
        # 此函数只用于观测值处。根据渗透率场和核心点坐标，给出经过映射后的H。
        # 输入：核心点位的坐标T_test_col, X_test_col, Y_test_col。
        # 输出：h_pred_pinn, TXY_all, hc_loss, hc_loss_, Hreal_, H_diff, Hreal
            # 产生含有ghost grid的全场坐标
            TXY_all_1, TXY_all, TXY_all_3, TXY_all_4, TXY_all_5, TXY_all_6 = HCP.generate_HCP_area(T_test_col.cpu().detach().numpy(), X_test_col.cpu().detach().numpy(), Y_test_col.cpu().detach().numpy())
            # 获得含有ghost grid的全场预测
            h_pred_pinn_1 =Net_pinn(TXY_all_1) # t-1
            h_pred_pinn = Net_pinn(TXY_all) # 核心点
            h_pred_pinn_3=Net_pinn(TXY_all_3) # x-1
            h_pred_pinn_4=Net_pinn(TXY_all_4) # x+1
            h_pred_pinn_5=Net_pinn(TXY_all_5) # y-1
            h_pred_pinn_6=Net_pinn(TXY_all_6) # y+1
            # H=torch.cat((h_pred_pinn_1_bc.data,     h_pred_pinn_bc.data,        h_pred_pinn_3_bc.data,\
            #             h_pred_pinn_4_bc.data,     h_pred_pinn_5_bc.data,      h_pred_pinn_6_bc.data),1)
            H=torch.cat((h_pred_pinn_1,     h_pred_pinn,        h_pred_pinn_3,\
                        h_pred_pinn_4,     h_pred_pinn_5,      h_pred_pinn_6),1)
            k_origin = k
            k_hc1 = np.vstack((np.vstack((np.zeros(len(k)), k)), np.zeros(len(k)))) # 增加无流条件对应的上下k=0
            k_hc2 = np.hstack((np.hstack((k_hc1[:,0].reshape(-1,1),k_hc1)), k_hc1[:,-1].reshape(-1,1)))
            # HCProjection
            A = HCP.buildA_batch(k_hc2, (torch.round(TXY_all[:, 1:2]*100/20-1)+1).long().cpu(),\
                    (torch.round(TXY_all[:, 2:3]*100/20-1)+1).long().cpu(), \
                    0.2,0.2,0.2, 1) # 坐标中的最后一个+1是因为此处对k进行了扩充（k_hc1和k_hc2），增加了ghost grid，所以原先下边界y=0处实际变成了y=1，同理，左边界x=0处变成了x=1
            # A = HCP.buildA_batch(k, (torch.round(TXY_all[:, 1:2]*100/20-1)).long().cpu(),\
            #         (torch.round(TXY_all[:, 2:3]*100/20-1)).long().cpu(), \
            #         0.2,0.2,0.2, 1)            
            H_, _, _, Hreal_ = HCP.projection_batch(A, H.unsqueeze(2))
            H_diff= torch.abs((H-H_)/H).mean()
            Hreal = H.unsqueeze(2)*(h_boun1_0-h_boun2_0)+h_boun2_0
            h_pred_pinn = torch.mm(H_, torch.tensor([[0],[1],[0],[0],[0],[0]]).type(torch.FloatTensor).to(device)) # 经过映射调整后的配点预测值                       
            return h_pred_pinn      
        if use_HCP == True:
            prediction = HCP_layer_observation (batch_x[:,0].reshape(-1,1), batch_x[:,1].reshape(-1,1), batch_x[:,2].reshape(-1,1), k)
        else:
            prediction=Net_pinn(batch_x)        
              
        # 预测配点或者配区域
        if use_HCP == True:
            prediction_f1=Net_pinn(batch_xf1)
            prediction_f2=Net_pinn(batch_xf2)
            prediction_f3=Net_pinn(batch_xf3)
            prediction_f4=Net_pinn(batch_xf4)
            prediction_f5=Net_pinn(batch_xf5)
            prediction_f6=Net_pinn(batch_xf6)
            # HCProjection
            A = HCP.buildA_batch(k, (torch.round(batch_xf2[:, 1:2]*100/20-1)).long().cpu(),\
                    (torch.round(batch_xf2[:, 2:3]*100/20-1)).long().cpu(), \
                    0.2,0.2,0.2, 1)
            H=torch.cat((prediction_f1.data,prediction_f2,prediction_f3.data,\
                        prediction_f4.data,prediction_f5.data,prediction_f6.data),1)
            H_, hc_loss, hc_loss_, Hreal_ = HCP.projection_batch(A, H.unsqueeze(2))
            hc_loss_set.append(torch.mean(torch.abs(hc_loss)))
            hc_loss_std_set.append(torch.std(torch.abs(hc_loss)))
            hc_loss_project_set.append(torch.mean(torch.abs(hc_loss_)))
            hc_loss_project_std_set.append(torch.std(torch.abs(hc_loss)))
            H_diff= torch.abs((H-H_)/H).mean()
            Hreal = H.unsqueeze(2)*(h_boun1_0-h_boun2_0)+h_boun2_0
            #print('mean: (Hreal)[:,1,:]: ', torch.abs((Hreal)[:,1,:]).mean())
            #print('mean: (Hreal_)[:,1,:]: ', torch.abs((Hreal_)[:,1,:]).mean())
            #print('mean: (Hreal - Hreal_)[:,1,:]: ', torch.abs((Hreal-Hreal_)[:,1,:]).mean())
            #print('max: (Hreal - Hreal_)[/H:,1,:]: ', (torch.abs((Hreal-Hreal_)[:,1,:])/torch.abs(Hreal[:,1,:])).max())
            prediction_f = torch.mm(H_, torch.tensor([[0],[1],[0],[0],[0],[0]]).type(torch.FloatTensor).to(device)) # 经过映射调整后的配点预测值
            #print('mean: (H - H_)/H[:,1,:]:',torch.abs(prediction_f - prediction_f2.permute(1,0)).mean())
            # if (epoch+1)%100 == 0:
            #     if ite == 0:
            #         np.savetxt('result_epoch_%d_H.csv'%(epoch+1),np.array(H.detach().cpu()).reshape(-1,6), delimiter=',')
            #         np.savetxt('result_epoch_%d_H_.csv'%(epoch+1),np.array(H_.detach().cpu()).reshape(-1,6), delimiter=',')        
        else:
            prediction_f = Net_pinn(batch_x2) # 直接PINN预测

      

        
        # 预测边界
        boun_pred=Net_pinn(TXY_boun_train) # 左边界，x=0.2处H=1
        boun2_pred=Net_pinn(TXY_boun2_train) # 右边界，x=10.2处H=0
        noflow_pred=Net_pinn(TXY_noflow)
        if use_HCP:
            noflow_pred_ghost=Net_pinn(TXY_noflow_ghost)
        def HCP_layer_boundary (boundary_type, direction, pressure, T_test_col, X_test_col, Y_test_col, k):
        # 此函数只用于边界条件处。根据渗透率场和核心点坐标，给出经过映射后的H。
        # 输入：boundary_type边界类型,0定压1无流； direction边界位置； pressure只用于定压边界；核心点位的坐标T_test_col, X_test_col, Y_test_col。
        # 输出：h_pred_pinn, TXY_all, hc_loss, hc_loss_, Hreal_, H_diff, Hreal
            # 产生含有ghost grid的全场坐标
            TXY_all_1, TXY_all, TXY_all_3, TXY_all_4, TXY_all_5, TXY_all_6 = HCP.generate_HCP_area(T_test_col.cpu().detach().numpy(), X_test_col.cpu().detach().numpy(), Y_test_col.cpu().detach().numpy())
            # 获得含有ghost grid的全场预测
            h_pred_pinn_1 =Net_pinn(TXY_all_1) # t-1
            h_pred_pinn = Net_pinn(TXY_all) # 核心点
            h_pred_pinn_3=Net_pinn(TXY_all_3) # x-1
            h_pred_pinn_4=Net_pinn(TXY_all_4) # x+1
            h_pred_pinn_5=Net_pinn(TXY_all_5) # y-1
            h_pred_pinn_6=Net_pinn(TXY_all_6) # y+1
            # 设置边界条件
            # 根据定压和无流条件，生成H场
            if boundary_type == 0: # 定压
                h_pred_pinn_1_bc, h_pred_pinn_bc, h_pred_pinn_3_bc, h_pred_pinn_4_bc, h_pred_pinn_5_bc, h_pred_pinn_6_bc = HCP.H_set_boundary_pressure(direction, pressure, h_pred_pinn_1, h_pred_pinn, h_pred_pinn_3, h_pred_pinn_4, h_pred_pinn_5, h_pred_pinn_6)
            if boundary_type == 1: # 无流
                h_pred_pinn_1_bc, h_pred_pinn_bc, h_pred_pinn_3_bc, h_pred_pinn_4_bc, h_pred_pinn_5_bc, h_pred_pinn_6_bc = HCP.H_set_boundary_noflow(direction, h_pred_pinn_1, h_pred_pinn, h_pred_pinn_3, h_pred_pinn_4, h_pred_pinn_5, h_pred_pinn_6)
            # H=torch.cat((h_pred_pinn_1_bc.data,     h_pred_pinn_bc.data,        h_pred_pinn_3_bc.data,\
            #             h_pred_pinn_4_bc.data,     h_pred_pinn_5_bc.data,      h_pred_pinn_6_bc.data),1)
            H=torch.cat((h_pred_pinn_1_bc,     h_pred_pinn_bc,        h_pred_pinn_3_bc,\
                        h_pred_pinn_4_bc,     h_pred_pinn_5_bc,      h_pred_pinn_6_bc),1)
            # 根据定压和无流条件，生成k场
            k_origin = k
            k_hc1 = np.vstack((np.vstack((np.zeros(len(k)), k)), np.zeros(len(k)))) # 增加无流条件对应的上下k=0
            k_hc2 = np.hstack((np.hstack((k_hc1[:,0].reshape(-1,1),k_hc1)), k_hc1[:,-1].reshape(-1,1)))
            # HCProjection
            A = HCP.buildA_batch(k_hc2, (torch.round(TXY_all[:, 1:2]*100/20-1)+1).long().cpu(),\
                    (torch.round(TXY_all[:, 2:3]*100/20-1)+1).long().cpu(), \
                    0.2,0.2,0.2, 1) # 坐标中的最后一个+1是因为此处对k进行了扩充（k_hc1和k_hc2），增加了ghost grid，所以原先下边界y=0处实际变成了y=1，同理，左边界x=0处变成了x=1
            H_, _, _, Hreal_ = HCP.projection_batch(A, H.unsqueeze(2))
            H_diff= torch.abs((H-H_)/H).mean()
            Hreal = H.unsqueeze(2)*(h_boun1_0-h_boun2_0)+h_boun2_0
            h_pred_pinn = torch.mm(H_, torch.tensor([[0],[1],[0],[0],[0],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
            h_pred_pinn_5 = torch.mm(H_, torch.tensor([[0],[0],[0],[0],[1],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
            h_pred_pinn_6 = torch.mm(H_, torch.tensor([[0],[0],[0],[0],[0],[1]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
            # return h_pred_pinn, h_pred_pinn_5, h_pred_pinn_6, TXY_all, hc_loss, hc_loss_, Hreal_, H_diff, Hreal
            return h_pred_pinn, h_pred_pinn_5, h_pred_pinn_6
        # boun_pred, _, _ = HCP_layer_boundary (0, 0, 1, TXY_boun_train[:,0].reshape(-1,1), TXY_boun_train[:,1].reshape(-1,1), TXY_boun_train[:,2].reshape(-1,1), k)
        # boun2_pred, _, _ = HCP_layer_boundary (0, 1, 0, TXY_boun2_train[:,0].reshape(-1,1), TXY_boun2_train[:,1].reshape(-1,1), TXY_boun2_train[:,2].reshape(-1,1), k)
        # noflow_pred1, noflow_pred1_ghost, _ = HCP_layer_boundary (1, 0, -999, TXY_noflow1[:,0].reshape(-1,1), TXY_noflow1[:,1].reshape(-1,1), TXY_noflow1[:,2].reshape(-1,1), k)
        # noflow_pred2, _, noflow_pred2_ghost = HCP_layer_boundary (1, 1, -999, TXY_noflow2[:,0].reshape(-1,1), TXY_noflow2[:,1].reshape(-1,1), TXY_noflow2[:,2].reshape(-1,1), k)
        # noflow_pred = torch.cat((noflow_pred1, noflow_pred2), 0)
        # noflow_pred_ghost = torch.cat((noflow_pred1_ghost, noflow_pred2_ghost), 0)

        # 初始实现
        ic_pred=Net_pinn(TXY_ic_train)
        ic2_pred=Net_pinn(TXY_ic2_train)

        # 配点处自动微分
        if use_HCP:
            H_grad = torch.autograd.grad(outputs=prediction_f.sum(), inputs=batch_xf2, create_graph=True)[0]
        else:
            H_grad = torch.autograd.grad(outputs=prediction_f.sum(), inputs=batch_x2, create_graph=True)[0]
        Ht = H_grad[:, 0].contiguous().view(n_train, 1)
        Hx = H_grad[:, 1].contiguous().view(n_train, 1)
        Hy = H_grad[:, 2].contiguous().view(n_train, 1) 
        if use_HCP:
            Hxx=torch.autograd.grad(outputs=Hx.sum(), inputs=batch_xf2,\
                                    create_graph=True)[0][:,1].contiguous().view(n_train, 1)
            Hyy=torch.autograd.grad(outputs=Hy.sum(), inputs=batch_xf2,\
                                    create_graph=True)[0][:,2].contiguous().view(n_train, 1)  
        else:
            Hxx=torch.autograd.grad(outputs=Hx.sum(), inputs=batch_x2,\
                                    create_graph=True)[0][:,1].contiguous().view(n_train, 1)
            Hyy=torch.autograd.grad(outputs=Hy.sum(), inputs=batch_x2,\
                                    create_graph=True)[0][:,2].contiguous().view(n_train, 1)                   
        # 无流边界微分    
        H_noflow_grad = torch.autograd.grad(outputs=noflow_pred.sum(), inputs=TXY_noflow, create_graph=True)[0]
        Hy_noflow=H_noflow_grad[:,2:3].contiguous()
        # 对比差分梯度与自动微分差异        
        if use_HCP == True:
            gradient_diff, diff_t_ID, diff_x_ID, diff_y_ID = HCP.compare_gradient(H, Ht, Hx, Hy)
            diff0, diff1, diff2, diff, pde012, a012, hc_loss_3_item = HCP.compare_loss (H, k, batch_xf2, Ht, Hxx, Hx, Hyy, Hy, K, k_x_, k_y_ )
        # 计算Loss
        f1=torch.pow((prediction-batch_y),2).mean()
        if use_HCP == True:
            f2=torch.pow(hc_loss_3_item.type(torch.FloatTensor),2).mean()
            #f2=torch.pow(hc_loss_3,2).mean()  # AH
            f2_=torch.pow(hc_loss_,2).mean() 
        f2_pde=torch.pow((Ss*Ht-K*Hxx-K*Hyy-k_x_*Hx-k_y_*Hy)*1,2).mean()      
        f3=torch.pow((boun_pred-H_boun_train),2).mean()+torch.pow((boun2_pred-H_boun2_train),2).mean()
        f6=torch.pow((ic_pred-H_ic_train),2).mean()+torch.pow((ic2_pred-H_ic2_train),2).mean()
        if use_HCP == True:
            f7_hc=25*torch.pow(torch.abs(noflow_pred-noflow_pred_ghost)*10,2).mean() #基于差分的无流边界条件
        else:
            f7=torch.pow(Hy_noflow*10,2).mean()
        #f7=torch.pow(Hy_noflow*10,2).mean()
        # 组合训练用loss 
        ################################################################################
        ################################################################################
        if use_HCP == True:
            loss=1*f1+100*f2_pde+1*f3+1*f6+1*f7_hc  
        else:
            if use_ANN == True:
                loss=1*f1
            else:
                loss=1*f1+100*f2_pde+1*f3+1*f6+1*f7
        ################################################################################
        ################################################################################
        # 更新网络
        loss.backward()
        optimizer2.step()
        # 打印各个loss
        print('f1:', f1)
        if use_HCP == True:
            print('f2:', f2)
            print('h_diff:', H_diff)
        print('f2_pde', f2_pde)
        print('f3:', f3)
        print('f6:', f6)
        if use_HCP:
            print('f7_hc:', f7_hc)
        else:
            print('f7:', f7)
        loss=loss.data
        f1=f1.data
        if use_HCP == True:
            f2=f2.data
            f2_=f2_.data
            H_diff=H_diff.data
        f2_pde=f2_pde.data
        f3=f3.data
        f6=f6.data
        if use_HCP:
            f7_hc=f7_hc.data
        else:
            f7=f7.data
        loss_pinn_set.append(loss)
        f1_set.append(f1)
        if use_HCP == True:
            f2_set.append(f2)
            f2updated_set.append(f2_)
            h_diff_set.append(H_diff)
            diff0_set.append(diff0)
            diff1_set.append(diff1)
            diff2_set.append(diff2)
            pde012_set.append(pde012)
            a012_set.append(a012)
        f2_pde_set.append(f2_pde)
        f3_set.append(f3)
        f6_set.append(f6)
        if use_HCP:
            f7_hc_set.append(f7_hc)
        else:
            f7_set.append(f7)
        if use_HCP == True:
            gradient_diff_set.append(gradient_diff)
        print('Epoch: ', epoch, '| Step: ', ite, '|loss: ',loss)
# np.savetxt('result_PDE.csv',np.array(f2_pde_set).reshape(-1,1), delimiter=',')
# np.savetxt('result_HC.csv',np.array(f2_set).reshape(-1,1), delimiter=',')


elapsed = time.time() - start_time                
print('Training time: %.4f' % (elapsed))

def HCP_layer(T_test_col, X_test_col, Y_test_col, k):
# 根据渗透率场和核心点坐标，给出经过映射后，符合边界条件的H。
# 注意！此时核心点坐标的排布顺序是有要求的，不是任意顺序的坐标点都可以（因为HCP.H_set_boundary函数对于坐标点排布顺序有要求）
# 输入：核心点位的坐标T_test_col, X_test_col, Y_test_col。
# 输出：h_pred_pinn, TXY_all, hc_loss, hc_loss_, Hreal_, H_diff, Hreal
    # 产生含有ghost grid的全场坐标
    TXY_all_1, TXY_all, TXY_all_3, TXY_all_4, TXY_all_5, TXY_all_6 = HCP.generate_HCP_area(T_test_col, X_test_col, Y_test_col)
    # 获得含有ghost grid的全场预测
    h_pred_pinn_1 =Net_pinn(TXY_all_1) # t-1
    h_pred_pinn = Net_pinn(TXY_all) # 核心点
    h_pred_pinn_3=Net_pinn(TXY_all_3) # x-1
    h_pred_pinn_4=Net_pinn(TXY_all_4) # x+1
    h_pred_pinn_5=Net_pinn(TXY_all_5) # y-1
    h_pred_pinn_6=Net_pinn(TXY_all_6) # y+1
    # 设置边界条件
    # 根据定压和无流条件，生成H场
    h_pred_pinn_1_bc, h_pred_pinn_bc, h_pred_pinn_3_bc, h_pred_pinn_4_bc, h_pred_pinn_5_bc, h_pred_pinn_6_bc = HCP.H_set_boundary(h_pred_pinn_1, h_pred_pinn, h_pred_pinn_3, h_pred_pinn_4, h_pred_pinn_5, h_pred_pinn_6)
    H=torch.cat((h_pred_pinn_1_bc.data,     h_pred_pinn_bc.data,        h_pred_pinn_3_bc.data,\
                 h_pred_pinn_4_bc.data,     h_pred_pinn_5_bc.data,      h_pred_pinn_6_bc.data),1)
    # 根据定压和无流条件，生成k场
    k_origin = k
    k_hc1 = np.vstack((np.vstack((np.zeros(len(k)), k)), np.zeros(len(k)))) # 增加无流条件对应的上下k=0
    k_hc2 = np.hstack((np.hstack((k_hc1[:,0].reshape(-1,1),k_hc1)), k_hc1[:,-1].reshape(-1,1)))
    # HCProjection
    A = HCP.buildA_batch(k_hc2, (torch.round(TXY_all[:, 1:2]*100/20-1)+1).long().cpu(),\
            (torch.round(TXY_all[:, 2:3]*100/20-1)+1).long().cpu(), \
            0.2,0.2,0.2, 1) # 坐标中的最后一个+1是因为此处对k进行了扩充（k_hc1和k_hc2），增加了ghost grid，所以原先下边界y=0处实际变成了y=1，同理，左边界x=0处变成了x=1
    H_, hc_loss, hc_loss_, Hreal_ = HCP.projection_batch(A, H.unsqueeze(2))
    H_diff= torch.abs((H-H_)/H).mean()
    Hreal = H.unsqueeze(2)*(h_boun1_0-h_boun2_0)+h_boun2_0
    h_pred_pinn_1 = torch.mm(H_, torch.tensor([[1],[0],[0],[0],[0],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
    h_pred_pinn = torch.mm(H_, torch.tensor([[0],[1],[0],[0],[0],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
    h_pred_pinn_3 = torch.mm(H_, torch.tensor([[0],[0],[1],[0],[0],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
    h_pred_pinn_4 = torch.mm(H_, torch.tensor([[0],[0],[0],[1],[0],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
    h_pred_pinn_5 = torch.mm(H_, torch.tensor([[0],[0],[0],[0],[1],[0]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
    h_pred_pinn_6 = torch.mm(H_, torch.tensor([[0],[0],[0],[0],[0],[1]]).type(torch.FloatTensor).to(device)) #此处是经过映射后的对应位置的结果
    # 设置边界条件
    _, h_pred_pinn, _, _, _, _ = HCP.H_set_boundary(h_pred_pinn_1, h_pred_pinn, h_pred_pinn_3, h_pred_pinn_4, h_pred_pinn_5, h_pred_pinn_6)
    return h_pred_pinn, TXY_all, hc_loss, hc_loss_, Hreal_, H_diff, Hreal

# PINN predict
H_all=hh.flatten()[:,None]
# 给出核心点坐标（整个流场所有时间步的）
T_test_col=T.flatten()[:,None]
X_test_col=X.flatten()[:,None]
Y_test_col=Y.flatten()[:,None] 
if use_HCP == True:
    h_pred_pinn, TXY_all, hc_loss, hc_loss_, Hreal_, H_diff, Hreal = HCP_layer(T_test_col, X_test_col, Y_test_col, k)
else:
    TXY_all = np.hstack((T_test_col,X_test_col,Y_test_col)) 
    TXY_all = Variable(torch.from_numpy(TXY_all).type(torch.FloatTensor), requires_grad=True).to(device)
    h_pred_pinn = Net_pinn(TXY_all)
h_pred_pinn = (h_pred_pinn*(h_boun1_0-h_boun2_0)+h_boun2_0).cpu().data.numpy() 
TXY_all=TXY_all.cpu()
h_array_pred_pinn=h_pred_pinn.reshape(nt,nx,ny)
#  calculate error
if config.use_noise == True:
    hh_origion = simu.hh_origion-200
    error_l2 = np.linalg.norm(hh_origion.flatten()-h_array_pred_pinn.flatten(),2)/np.linalg.norm(hh_origion.flatten(),2)
    print('Error L2: %e' % (error_l2))
    R2=1-np.sum((hh_origion.flatten()-h_array_pred_pinn.flatten())**2)/np.sum((hh_origion.flatten()-hh_origion.flatten().mean())**2)
    print('coefficient of determination  R2: %e' % (R2))
    # # 绘制交汇图  
    # noise = config.noise    
    # noise_level = '%d'%(noise*100)    
    # hh_all = np.array(dataset.H_train)
    # hh_origion_all = dataset.H_train_origion.flatten()
    # # hh_origion_all = (hh_origion_all)/2
    # # idx2 = np.random.choice(range(hh_all.shape[0]), 1000, replace=False)
    # plt.figure(figsize=(8,8))
    # plt.scatter(hh_origion_all,hh_all,marker='o',edgecolors='b')
    # plt.xlabel('Reference',fontsize=20)
    # plt.ylabel('Noisy observation',fontsize=20)
    # plt.title('Noise level '+noise_level+'%',fontsize=20)
    # plt.xlim(-0.1,1+0.1)
    # plt.ylim(-0.3,1+0.3)
    # plt.savefig('Noise level 555555 1800 normal %d.png'%(noise*100), dpi=600)
    # plt.show()
else:
    error_l2 = np.linalg.norm(hh.flatten()-h_array_pred_pinn.flatten(),2)/np.linalg.norm(hh.flatten(),2)
    print('Error L2: %e' % (error_l2))
    R2=1-np.sum((hh.flatten()-h_array_pred_pinn.flatten())**2)/np.sum((hh.flatten()-hh.flatten().mean())**2)
    print('coefficient of determination  R2: %e' % (R2))


                                                ##############

                                        ###############################

                            #######################################################

                    #########################################################################

##################################################################################################################
#绘图
##################################################################################################################
# 目录
# path = "\\4_no_label_parameter_results\\"#切换工作目录
# now = os.getcwd()# 查看当前工作目录
# os.chdir( now+path )# 修改当前工作目录
# new_path = os.getcwd()# 查看修改后的工作目录
# if use_HCP:
#     np.savetxt('result_lastepoch_H.csv',np.array(H.detach().cpu()).reshape(-1,6), delimiter=',')
#     np.savetxt('result_lastepoch_H_.csv',np.array(H_.detach().cpu()).reshape(-1,6), delimiter=',')
#     np.savetxt('result_AH.csv',np.array(hc_loss_set).reshape(-1,1), delimiter=',')
#     np.savetxt('result_AH_.csv',np.array(hc_loss_project_set).reshape(-1,1), delimiter=',')
#     np.savetxt('result_AH_std.csv',np.array(hc_loss_std_set).reshape(-1,1), delimiter=',')
#     np.savetxt('result_AH_std_.csv',np.array(hc_loss_project_std_set).reshape(-1,1), delimiter=',')
#     np.savetxt('result_gradient.csv',gradient_diff_set, delimiter=',')

# # 画局部取值图，看梯度差分计算是否合理
# def NN_value_surface (t, x, y, n=1, n_unit = 10):
# # 功能：计算某一个区域内差分尺度范围的神经网络取值面，可以对比对应配点附近临域自动微分的梯度和配区域差分的梯度是否相似
# # 输入：txy对应配点坐标，n决定绘图区域的长与宽（2n倍的差分间隔），n_unit对应于绘图的精细度（差分间隔分为n_unit份）
# # 输出：value_surface一个尺寸为（2n/delta， 2n/delta）的矩阵，元素为对应位置的神经网预测值
#     x_min = x-n*0.2
#     x_max = x+n*0.2
#     y_min = y-n*0.2
#     y_max = y+n*0.2
#     len_area = 2*n*n_unit+1
#     coordinate = np.zeros((np.int(len_area*len_area), 3))
#     coordinate[:,0] = t
#     coordinate[:,1] = np.tile(np.arange(x_min,x_max, 0.2/n_unit), np.int(len_area))
#     coordinate[:,2] = np.arange(x_min,x_max,0.2/n_unit).repeat(len_area, axis=0)
#     coordinate = Variable(torch.from_numpy(coordinate).type(torch.FloatTensor),requires_grad=True).to(device)
#     prediction_area = Net_pinn(coordinate)
#     prediction_area_square = prediction_area.cpu().detach().numpy().reshape((len_area, len_area))
#     return prediction_area_square
# if config.plot_value_surface == True:
#     for i in range (3):
#         for j in range (2):
#             t_plot = i*20+4
#             y_plot = j*20+14
#             n_unit=50
#             Z_plot =  NN_value_surface (t_plot, 5, y_plot, n=1, n_unit=n_unit)
#             X_plot=np.arange(5-0.2, 5+0.2001, 0.2/n_unit) 
#             Y_plot=np.arange(y_plot-0.2, y_plot+0.2001, 0.2/n_unit)
#             fig = plt.figure()
#             from mpl_toolkits.mplot3d import Axes3D
#             ax = Axes3D(fig)
#             X_plot,Y_plot=np.meshgrid(X_plot,Y_plot)   
#             plt.title('Time step %d'%(t_plot+1),fontsize=10)
#             ax.plot_surface(X_plot, Y_plot, Z_plot, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))

# # #画loss图

# plt.figure()     
# plt.plot(range(len(loss_pinn_set)),loss_pinn_set)
# plt.xlabel('Iteration')
# plt.ylabel('Loss')
# loss_result_loss = np.array(loss_pinn_set).reshape(-1,1)
# np.savetxt('result_loss.csv',loss_result_loss, delimiter=',')
# plt.savefig('result_loss.png', dpi=600)

# plt.figure()   
# plt.plot(range(len(f1_set)),f1_set)
# plt.xlabel('Iteration')
# plt.ylabel('Observation loss')
# #plt.ylim([0,0.001])
# loss_result_f1=np.array(f1_set).reshape(-1,1)
# loss_result_all=np.hstack((loss_result_loss,loss_result_f1))
# np.savetxt('result_f1.csv',loss_result_f1, delimiter=',')
# plt.savefig('result_f1.png', dpi=600)

# if use_HCP == True:
#     plt.figure()     
#     plt.plot(range(len(f2_set)),f2_set)      
#     plt.xlabel('Iteration')
#     plt.ylabel('HCP loss')    
#     #plt.ylim([0,0.001])
#     loss_result_f2AH=np.array(f2_set).reshape(-1,1)
#     loss_result_all=np.hstack((loss_result_all,loss_result_f2AH))
#     np.savetxt('result_f2AH.csv',loss_result_f2AH, delimiter=',')
#     plt.savefig('result_f2AH.png', dpi=600)
    

#     plt.figure()     
#     plt.plot(range(len(f2updated_set)),f2updated_set)      
#     plt.xlabel('Iteration')
#     plt.ylabel('Projected HCP loss')
#     loss_result_f2updatedAH=np.array(f2updated_set).reshape(-1,1)
#     loss_result_all=np.hstack((loss_result_all,loss_result_f2updatedAH))
#     np.savetxt('result_f2updatedAH_.csv',loss_result_f2updatedAH, delimiter=',')  
#     plt.savefig('result_f2updatedAH.png', dpi=600)

#     plt.figure()     
#     plt.plot(range(len(h_diff_set)),h_diff_set)     
#     plt.xlabel('Iteration')
#     plt.ylabel('HCP change ratio') 
#     #plt.ylim([0,0.005])  
#     np.savetxt('result_h_diff.csv',np.array(h_diff_set).reshape(-1,1), delimiter=',')  
#     plt.savefig('result_h_diff.png', dpi=600)

#     plt.figure()     
#     plt.plot(range(len(diff0_set)),diff0_set, label='a0-pde0')   
#     plt.plot(range(len(diff1_set)),diff1_set, label='a1-pde1')
#     plt.plot(range(len(diff2_set)),diff2_set, label='a2-pde2')  
#     plt.xlabel('Iteration')
#     plt.ylabel('diff:(a0-pde0)(a1-pde1)(a2-pde2)') 
#     plt.legend()

#     plt.figure()     
#     plt.plot(range(len(pde012_set)),pde012_set)   
#     plt.xlabel('Iteration')
#     plt.ylabel('pde012') 
#     plt.legend()

#     plt.figure()     
#     plt.plot(range(len(a012_set)),a012_set)   
#     plt.xlabel('Iteration')
#     plt.ylabel('a012') 
#     plt.legend()
# plt.figure()     
# plt.plot(range(len(f2_pde_set)),f2_pde_set)      
# plt.xlabel('Iteration')
# plt.ylabel('PDE loss')    
# #plt.ylim([0,0.001])
# loss_result_f2pde=np.array(f2_pde_set).reshape(-1,1)
# loss_result_all=np.hstack((loss_result_all,loss_result_f2pde))
# np.savetxt('result_f2pde.csv',loss_result_f2pde, delimiter=',')
# plt.savefig('result_f2pde.png', dpi=600)
# if use_HCP == True:   
#     plt.figure(figsize=(5,5))
#     plt.scatter(np.array(f2_pde_set),np.array(f2_set),marker='o',edgecolors='b')
#     plt.xlabel('pde',fontsize=18)
#     plt.ylabel('hc',fontsize=18)
#     plt.title("PDE v.s. HCP",fontsize=18)

# plt.figure()     
# plt.plot(range(len(f3_set)),f3_set)      
# plt.xlabel('Iteration')
# plt.ylabel('Constant pressure boundary loss')  
# #plt.ylim([0,0.05])
# loss_result_f3=np.array(f3_set).reshape(-1,1)
# loss_result_all=np.hstack((loss_result_all,loss_result_f3))
# np.savetxt('result_f3.csv',loss_result_f3, delimiter=',')
# plt.savefig('result_f3.png', dpi=600)

# '''
# plt.figure()     
# plt.plot(range(len(f4_set)),f4_set)      
# plt.xlabel('Iteration')
# plt.ylabel('f4_loss')  
# plt.ylim([0,0.05])
# plt.figure()     
# plt.plot(range(len(f5_set)),f5_set)      
# plt.xlabel('Iteration')
# plt.ylabel('f5_loss')  
# plt.ylim([0,0.05])
# '''

# plt.figure()     
# plt.plot(range(len(f6_set)),f6_set)      
# plt.xlabel('Iteration')
# plt.ylabel('Initial condition loss') 
# loss_result_f6=np.array(f6_set).reshape(-1,1)
# loss_result_all=np.hstack((loss_result_all,loss_result_f6))
# np.savetxt('result_f6.csv',loss_result_f6, delimiter=',')
# plt.savefig('result_f6.png', dpi=600)
# #plt.ylim([0,0.05])
# # plt.figure()     
# # plt.plot(range(len(f7_set)),f7_set)      
# # plt.xlabel('Iteration')
# # plt.ylabel('f7_loss') 
# #plt.ylim([0,0.05])
# if use_HCP:
#     plt.figure()     
#     plt.plot(range(len(f7_hc_set)),f7_hc_set)      
#     plt.xlabel('Iteration')
#     plt.ylabel('No flow boundary loss') 
#     loss_result_f7=np.array(f7_hc_set).reshape(-1,1)
#     loss_result_all=np.hstack((loss_result_all,loss_result_f7))
#     #plt.ylim([0,0.005])
#     np.savetxt('result_f7_hc.csv',loss_result_f7, delimiter=',')
#     plt.savefig('result_f7_hc.png', dpi=600)
# else:
#     plt.figure()     
#     plt.plot(range(len(f7_set)),f7_set)      
#     plt.xlabel('Iteration')
#     plt.ylabel('No flow boundary loss') 
#     loss_result_f7=np.array(f7_set).reshape(-1,1)
#     loss_result_all=np.hstack((loss_result_all,loss_result_f7))
#     #plt.ylim([0,0.005])
#     np.savetxt('result_f7.csv',loss_result_f7, delimiter=',')
#     plt.savefig('result_f7.png', dpi=600)    
# # plt.figure(figsize=(5,5))
# # plt.scatter(np.array(f7_set),np.array(f7_hc_set),marker='o',edgecolors='b')
# # plt.xlabel('pde',fontsize=18)
# # plt.ylabel('hc',fontsize=18)
# # plt.title("pde v.s. hc",fontsize=18)
# np.savetxt('result_loss_all.csv',loss_result_all, delimiter=',')

# # PINN plot
# x=x*100
# y=y*100
# for i_t in range(nt-31,nt,10):      
#     # Row 0: h(t,x)     
#     plt.figure(figsize=(3,3))    
#     mm1=plt.imshow(hh[i_t], interpolation='nearest', cmap='rainbow', 
#                   extent=[x.min(), x.max(), y.min(),y.max()], 
#                   origin='lower')
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.title('Reference $H$ at time step %d'%(i_t+1),fontsize=10)
#     plt.colorbar(mm1,fraction=0.046, pad=0.04)    
#     line = np.linspace(x.min(), x.max(), 2)[:,None]
#     plt.plot(line, y[15]*np.ones((2,1)), 'w-', linewidth = 1)
#     plt.plot(line, y[30]*np.ones((2,1)), 'w-', linewidth = 1)
#     plt.plot(line, y[45]*np.ones((2,1)), 'w-', linewidth = 1)
#     plt.savefig('Reference $H$ at time step %d .png'%(i_t+1), dpi=600)
#     # Row 1: pred h(t,x) 
#     plt.figure(figsize=(3,3))
#     mm2=plt.imshow(h_array_pred_pinn[i_t], interpolation='nearest', cmap='rainbow', 
#                   extent=[x.min(), x.max(), y.min(),y.max()], 
#                   origin='lower')
#     plt.xlabel('$x$')
#     plt.ylabel('$y$')
#     plt.title('Predicted $H$ at time step %d'%(i_t+1),fontsize=10)
#     plt.colorbar(mm2,fraction=0.046, pad=0.04)
#     line = np.linspace(x.min(), x.max(), 2)[:,None]
#     plt.plot(line, y[15]*np.ones((2,1)), 'w-', linewidth = 1)
#     plt.plot(line, y[30]*np.ones((2,1)), 'w-', linewidth = 1)
#     plt.plot(line, y[45]*np.ones((2,1)), 'w-', linewidth = 1)  
#     plt.savefig('Predicted $H$ at time step %d .png'%(i_t+1), dpi=600)   
#     # Row 2: u(t,x) slices 
#     plt.figure(figsize=(10,5))
#     gs1 = gridspec.GridSpec(1, 3)
#     # gs1.update(top=-0.05, bottom=-0.5, left=0.1, right=0.9, wspace=0.6)
#     # 320处
#     ax = plt.subplot(gs1[0, 0])
#     ax.plot(x,hh[i_t,15,:], 'b-', linewidth = 2, label = 'Reference')       
#     ax.plot(x,h_array_pred_pinn[i_t,15,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     ax.set_ylabel('$H$')    
#     ax.set_title('$y = 320$', fontsize = 10)
#     ax.set_xlim([min(x)-0.1*max(x),max(x)+0.1*max(x)])
#     # 620处
#     ax = plt.subplot(gs1[0, 1])
#     ax.plot(x,hh[i_t,30,:], 'b-', linewidth = 2, label = 'Reference')       
#     ax.plot(x,h_array_pred_pinn[i_t,30,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     #ax.set_ylabel('$H$')
#     ax.set_xlim([min(x)-0.1*max(x),max(x)+0.1*max(x)])
#     ax.set_title('$y = 620$', fontsize = 10)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
#     # 920处
#     ax = plt.subplot(gs1[0, 2])
#     ax.plot(x,hh[i_t,45,:], 'b-', linewidth = 2, label = 'Reference')       
#     ax.plot(x,h_array_pred_pinn[i_t,45,:], 'r--', linewidth = 2, label = 'Prediction')
#     ax.set_xlabel('$x$')
#     #ax.set_ylabel('$H$')
#     ax.set_xlim([min(x)-0.1*max(x),max(x)+0.1*max(x)])
#     ax.set_title('$y = 920$', fontsize = 10)
#     plt.savefig('Slices $H$ at time step %d .png'%(i_t+1), dpi=600)
#     # plt.show()
#     slice_h = np.hstack((hh[i_t,15,:].reshape(-1,1), hh[i_t,30,:].reshape(-1,1)))
#     slice_h = np.hstack((slice_h, hh[i_t,45,:].reshape(-1,1)))
#     slice_h = np.hstack((slice_h, h_array_pred_pinn[i_t,15,:].reshape(-1,1)))
#     slice_h = np.hstack((slice_h, h_array_pred_pinn[i_t,30,:].reshape(-1,1)))
#     slice_h = np.hstack((slice_h, h_array_pred_pinn[i_t,45,:].reshape(-1,1)))
#     np.savetxt('result_slice_h_%d.csv'%(i_t+1),slice_h, delimiter=',')

# lastepoch_prediction = h_array_pred_pinn[-1,:,:]
# lastepoch_groundtruth = hh[-1,:,:]
# np.savetxt('result_lastepoch_prediction.csv',lastepoch_prediction, delimiter=',')
# np.savetxt('result_lastepoch_groundtruth.csv',lastepoch_groundtruth, delimiter=',')
    
# time30_prediction = h_array_pred_pinn[-20,:,:]
# time30_groundtruth = hh[-20,:,:]
# np.savetxt('result_lastepoch_time30_prediction.csv',time30_prediction, delimiter=',')
# np.savetxt('result_lastepoch_time30_groundtruth.csv',time30_groundtruth, delimiter=',')
   
# # 绘制交汇图   
# idx2 = np.random.choice(range(H_all.shape[0]), 300, replace=False)
# plt.figure(figsize=(5,5))
# plt.scatter(H_all[idx2],h_pred_pinn[idx2],marker='o',edgecolors='b')
# plt.xlabel('Reference',fontsize=18)
# plt.ylabel('Prediction',fontsize=18)
# if use_HCP == True:
#     plt.title("Results Comparison for HCP",fontsize=18)
# else:
#     if use_ANN == True:
#         plt.title("Results Comparison for ANN",fontsize=18)
#     else:
#         plt.title("Results Comparison for TgNN",fontsize=18)
# plt.xlim(h_boun2_0-0.2,h_boun1_0+0.2)
# plt.ylim(h_boun2_0-0.2,h_boun1_0+0.2)
# plt.savefig('Results Comparison for TgNN.png', dpi=600)

# # 绘制渗透率分布
# plt.figure(figsize=(3,3))
# mm=plt.imshow(logk,origin='lower',cmap='jet',interpolation='nearest',aspect='equal',
#            extent=[0, 51, 0,51])
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.title("ln$K(x,y)$")
# plt.xlim(0,51)
# plt.ylim(0,51)
# plt.colorbar(mm,fraction=0.046, pad=0.04)
# my_x_ticks = np.arange(0,51, 10)
# my_y_ticks = np.arange(0,51, 10)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# plt.savefig('K.png', dpi=600)

# plt.show()

print('HCP Mission complete')