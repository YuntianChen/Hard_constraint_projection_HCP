import numpy as np
import configure as config
import simulation as simu
from pyDOE import lhs
import torch
from torch.autograd import Variable

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

hh = simu.hh-200
N_h = config.N_h
h_boun1_0 = simu.h_boun1_0-200
h_boun2_0 = simu.h_boun2_0-200
k = simu.k
x = simu.x
y = simu.y
t = simu.t
h_boun1 = simu.h_boun1
h_boun2 = simu.h_boun2
dx = simu.dx
dy = simu.dy
dt = simu.dt
w_x = simu.w_x
w_y = simu.w_y
lamda_xy = simu.lamda_xy
kesi = simu.kesi
n_eigen = simu.n_eigen
eta = simu.eta
L_x = simu.L_x
L_y = simu.L_y
mean_logk = simu.mean_logk

nx = config.nx     #网格个数
ny = config.ny
nt = config.nt      #时间步数
N_boun_hyper = config.N_boun_hyper
N_no_flow = config.N_no_flow
Nf_hyper = config.Nf_hyper
device = config.device
#########################################################
#定义数据空间
#########################################################
xx=np.empty((nt,nx,ny))
yy=np.empty((nt,nx,ny))
tt=np.empty((nt,nx,ny))
kk=np.empty((nt,nx,ny))
Y,T,X = np.meshgrid(y,t,x)
xx=X
yy=Y
tt=T
for i_t in range(nt):
    kk[i_t,:,:]=k[:,:]

#########################################################
#提取训练数据
#########################################################
t_train=18
H_col=hh[0,:,:].flatten()[:,None]
if config.use_noise == 1:
    H_col_origion=(simu.hh_origion-200)[0,:,:].flatten()[:,None]
T_col=tt[0,:,:].flatten()[:,None]
X_col=xx[0,:,:].flatten()[:,None]
Y_col=yy[0,:,:].flatten()[:,None]
K_col=kk[0,:,:].flatten()[:,None]
TXYK = np.hstack((T_col,X_col,Y_col,K_col))
TXY_all = np.hstack((T_col,X_col,Y_col))
H_all=H_col
idx = np.random.choice(TXYK.shape[0], N_h, replace=False)#抽样id
TXYK_train =TXYK[idx,:]
H_train = H_col[idx,:]  
if config.use_noise == True:
    H_train_origion = H_col_origion[idx,:]
for i in range(1,t_train): #上面记录第0步，此处拼接上第1到第t_train-1步
    H_col=hh[i,:,:].flatten()[:,None]
    if config.use_noise == 1:
        H_col_origion = (simu.hh_origion-200)[i,:,:].flatten()[:,None]
    T_col=tt[i,:,:].flatten()[:,None]
    X_col=xx[i,:,:].flatten()[:,None]
    Y_col=yy[i,:,:].flatten()[:,None]
    K_col=kk[i,:,:].flatten()[:,None]
    TXYK = np.hstack((T_col,X_col,Y_col,K_col))
    txy_all=np.hstack((T_col,X_col,Y_col))
    h_all=H_col    
    TXY_all=np.vstack((TXY_all,txy_all)) #拼接存储
    H_all=np.vstack((H_all,h_all)) #拼接存储
    idx = np.random.choice(TXYK.shape[0], N_h, replace=False)
    txyk_train =TXYK[idx,:]
    h_train = H_col[idx,:]
    H_train = np.vstack((H_train,h_train)) #拼接存储
    if config.use_noise == 1:
        h_train_origion = H_col_origion[idx,:]
        H_train_origion = np.vstack((H_train_origion, h_train_origion))
    TXYK_train =np.vstack((TXYK_train,txyk_train)) #拼接存储
    
for i in range(t_train,nt):# 此处拼接训练数据以外的步骤，此时已经没有观测值，因此没有YXYK_train和H_train
    H_col=hh[i,:,:].flatten()[:,None]
    T_col=tt[i,:,:].flatten()[:,None]
    X_col=xx[i,:,:].flatten()[:,None]
    Y_col=yy[i,:,:].flatten()[:,None]
    K_col=kk[i,:,:].flatten()[:,None]
    txy_all=np.hstack((T_col,X_col,Y_col))
    h_all=H_col    
    TXY_all=np.vstack((TXY_all,txy_all))
    H_all=np.vstack((H_all,h_all))
TXY_train=TXYK_train[:,0:3]
n_train0=len(H_train) 
H_train=(H_train-h_boun2_0)/(h_boun1_0-h_boun2_0) # normalization 
if config.use_noise == 1:
    H_train_origion=(H_train_origion-h_boun2_0)/(h_boun1_0-h_boun2_0) # normalization 

if config.use_outlier == True:
    n_outlier=round(n_train0*config.outlier_per)
    idx_outlier1 = np.random.choice(H_train.shape[0], round(n_outlier), replace=False)
    H_train_origion = H_train.copy()
    H_train[idx_outlier1]=np.random.uniform(1,2,H_train[idx_outlier1].shape)
    import matplotlib.pyplot as plt
    # # 绘制交汇图   
    # outlier_level = '%d'%(config.outlier_per*100) 
    
    # plt.figure(figsize=(8,8))
    # plt.scatter(H_train_origion,H_train, marker='o',edgecolors='b')
    # plt.xlabel('Reference',fontsize=20)
    # plt.ylabel('Outlier',fontsize=20)
    # plt.title('Outlier proportion '+outlier_level+'%',fontsize=20)
    # plt.xlim(-0.1,1+0.1)
    # plt.ylim(-0.1,2+0.1)
    # plt.savefig('Outlier proportion normal 888888 1800 %d.png'%(config.outlier_per*100), dpi=600)
    # plt.show()
    
#########################################################
#提取边界数据
#########################################################
# x=0.2处H=1
N_boun1=N_boun_hyper
X_boun_col=x[0]*np.ones((N_boun1,1))
Y_boun_col=y[0]+(y[50]-y[0])*lhs(1, N_boun1)
T_boun_col=t[0]+(t[49]-t[0])*lhs(1, N_boun1)
H_boun_col=h_boun1*np.ones((N_boun1,1))
TXY_boun = np.hstack((T_boun_col,X_boun_col,Y_boun_col))   
H_boun=H_boun_col
TXY_boun_train=TXY_boun
H_boun_train=H_boun
TXY_boun_train = torch.from_numpy(TXY_boun_train).type(torch.FloatTensor).to(device)
H_boun_train = torch.from_numpy(H_boun_train).type(torch.FloatTensor).to(device)
#提取边界数据2
# x=10.2处H=0
N_boun2=N_boun_hyper
X_boun2_col=x[nx-1]*np.ones((N_boun2,1))
Y_boun2_col=y[0]+(y[50]-y[0])*lhs(1, N_boun2)
T_boun2_col=t[0]+(t[49]-t[0])*lhs(1, N_boun2)
H_boun2_col=h_boun2*np.ones((N_boun2,1))
TXY_boun2 = np.hstack((T_boun2_col,X_boun2_col,Y_boun2_col))   
H_boun2=H_boun2_col
TXY_boun2_train=TXY_boun2
H_boun2_train=H_boun2
TXY_boun2_train = torch.from_numpy(TXY_boun2_train).type(torch.FloatTensor).to(device)
H_boun2_train = torch.from_numpy(H_boun2_train).type(torch.FloatTensor).to(device)

#########################################################
#提取初始条件数据
#########################################################
# 所有位置t=0时，H=0
N_ic=5000
X_ic_col=x[1]+(x[50]-y[1])*lhs(1, N_ic)
Y_ic_col=y[0]+(y[50]-y[0])*lhs(1, N_ic)
T_ic_col=np.zeros((N_ic,1))
H_ic_col=h_boun2*np.ones((N_ic,1))
TXY_ic = np.hstack((T_ic_col,X_ic_col,Y_ic_col))   
H_ic=H_ic_col
TXY_ic_train=TXY_ic
H_ic_train=H_ic
TXY_ic_train = torch.from_numpy(TXY_ic_train).type(torch.FloatTensor).to(device)
H_ic_train = torch.from_numpy(H_ic_train).type(torch.FloatTensor).to(device)
#提取初始条件数据2
# x=0.2,t=0时，H=0
N_ic2=5000
X_ic2_col=x[0]*np.ones((N_ic2,1))
Y_ic2_col=y[0]+(y[50]-y[0])*lhs(1, N_ic2)
T_ic2_col=np.zeros((N_ic2,1))
H_ic2_col=h_boun1*np.ones((N_ic2,1))
TXY_ic2 = np.hstack((T_ic2_col,X_ic2_col,Y_ic2_col))   
H_ic2=H_ic2_col
TXY_ic2_train=TXY_ic2
H_ic2_train=H_ic2
TXY_ic2_train = torch.from_numpy(TXY_ic2_train).type(torch.FloatTensor).to(device)
H_ic2_train = torch.from_numpy(H_ic2_train).type(torch.FloatTensor).to(device)

#########################################################
#提取无流边界数据
#########################################################
N_noflow1=N_no_flow
X_noflow1_col=x[0]+(x[50]-x[0])*lhs(1, N_noflow1)
Y_noflow1_col=y[0]*np.ones((N_noflow1,1))
Y_noflow1_col_ghost=Y_noflow1_col-0.2
T_noflow1_col=t[0]+(t[49]-t[0])*lhs(1, N_noflow1)
TXY_noflow1 = np.hstack((T_noflow1_col,X_noflow1_col,Y_noflow1_col))   
TXY_noflow1_ghost = np.hstack((T_noflow1_col,X_noflow1_col,Y_noflow1_col_ghost))  
#提取无流边界数据2
N_noflow2=N_no_flow
X_noflow2_col=x[0]+(x[50]-x[0])*lhs(1, N_noflow2)
Y_noflow2_col=y[50]*np.ones((N_noflow2,1))
Y_noflow2_col_ghost=Y_noflow2_col+0.2
T_noflow2_col=t[0]+(t[49]-t[0])*lhs(1, N_noflow2)
TXY_noflow2 = np.hstack((T_noflow2_col,X_noflow2_col,Y_noflow2_col))   
TXY_noflow2_ghost = np.hstack((T_noflow2_col,X_noflow2_col,Y_noflow2_col_ghost)) 
TXY_noflow=np.vstack((TXY_noflow1,TXY_noflow2))
TXY_noflow_ghost=np.vstack((TXY_noflow1_ghost,TXY_noflow2_ghost))
TXY_noflow= Variable(torch.from_numpy(TXY_noflow).type(torch.FloatTensor).to(device), requires_grad=True)
TXY_noflow_ghost= Variable(torch.from_numpy(TXY_noflow_ghost).type(torch.FloatTensor).to(device), requires_grad=True)
#########################################################
#提取配点数据
#########################################################
Nf=Nf_hyper
filter_point = filter(lambda mm: (mm[1]!=0.2) and (mm[1]!=10.2) and (mm[2]!=0.2) and (mm[2]!=10.2) and (mm[0]!=0.2) ,TXY_all)
TXY_filter=np.array(list(filter_point))
#TXY_filter=np.array(TXY_all)
T_all=TXY_filter[:,0:1]
X_all=TXY_filter[:,1:2]
Y_all=TXY_filter[:,2:3]
idx_t = np.random.choice(TXY_filter.shape[0], Nf, replace=False)
idx_x = np.random.choice(TXY_filter.shape[0], Nf, replace=False)
idx_y = np.random.choice(TXY_filter.shape[0], Nf, replace=False)
Tf=T_all[idx_t,:]
Xf=X_all[idx_x,:]
Yf=Y_all[idx_y,:]
TXY_f_train=np.hstack((Tf,Xf,Yf))
n_colloc=np.shape(TXY_f_train)[0]
# 针对硬约束的配区域数据
TXY_f_train1=TXY_f_train.copy()
TXY_f_train1[:,0:1]=TXY_f_train1[:,0:1]-dt
TXY_f_train2=TXY_f_train.copy()
TXY_f_train3=TXY_f_train.copy()
TXY_f_train3[:,1:2]=TXY_f_train3[:,1:2]-dx
TXY_f_train4=TXY_f_train.copy()
TXY_f_train4[:,1:2]=TXY_f_train4[:,1:2]+dx
TXY_f_train5=TXY_f_train.copy()
TXY_f_train5[:,2:3]=TXY_f_train5[:,2:3]-dy
TXY_f_train6=TXY_f_train.copy()
TXY_f_train6[:,2:3]=TXY_f_train6[:,2:3]+dy


#########################################################
# 训练网络的数据
#########################################################
# TXY_train
TXY_train = torch.from_numpy(TXY_train).type(torch.FloatTensor)
TXY_f_train = torch.from_numpy(TXY_f_train).type(torch.FloatTensor)
H_train = torch.from_numpy(H_train).type(torch.FloatTensor)
TXY_f_train1 = torch.from_numpy(TXY_f_train1).type(torch.FloatTensor)
TXY_f_train2 = torch.from_numpy(TXY_f_train2).type(torch.FloatTensor)
TXY_f_train3 = torch.from_numpy(TXY_f_train3).type(torch.FloatTensor)
TXY_f_train4 = torch.from_numpy(TXY_f_train4).type(torch.FloatTensor)
TXY_f_train5 = torch.from_numpy(TXY_f_train5).type(torch.FloatTensor)
TXY_f_train6 = torch.from_numpy(TXY_f_train6).type(torch.FloatTensor)

# 生成k_train和TXYK_kxky_f_train
w_x_tf = torch.from_numpy(w_x).type(torch.FloatTensor)
w_y_tf = torch.from_numpy(w_y).type(torch.FloatTensor)
lamda_xy_tf = torch.from_numpy(lamda_xy).type(torch.FloatTensor)
kesi = torch.from_numpy(kesi).type(torch.FloatTensor)
x_tf = Variable(TXY_f_train[:,1], requires_grad=True)
y_tf = Variable(TXY_f_train[:,2], requires_grad=True)
fx_train=simu.eigen_func2(n_eigen,w_x_tf,eta,L_x,x_tf)
fy_train=simu.eigen_func2(n_eigen,w_y_tf,eta,L_y,y_tf)
k_train=simu.k2(kesi,fx_train.transpose(0,1),fy_train.transpose(0,1),mean_logk,lamda_xy_tf.transpose(0,1))
k_x = torch.autograd.grad(outputs=k_train.sum(), inputs=x_tf, \
                          create_graph=True,allow_unused=True)[0].view(Nf, 1).detach()
k_y = torch.autograd.grad(outputs=k_train.sum(), inputs=y_tf, \
                          create_graph=True,allow_unused=True)[0].view(Nf, 1).detach()
k_train=k_train.detach().numpy().reshape(Nf,1)
TXYK_kxky_f_train =np.hstack((TXY_f_train,k_train,k_x,k_y))
TXYK_kxky_f_train = torch.from_numpy(TXYK_kxky_f_train).type(torch.FloatTensor)


