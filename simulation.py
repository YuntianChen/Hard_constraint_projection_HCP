import torch
import matplotlib.pyplot as plt
import os
import os.path
import configure as config
import numpy as np
import re
import operator

torch.random.manual_seed(config.seed)
np.random.seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

def search(f,a,h,n):
#%功能：找到发f(x)在区间[a,+∞)上所有隔根区间
#%输入：f(x):所求方程函数；[a,+∞):有根区间；h:步长；n:所需要区间个数
#%输出：隔根区间[c,d] 
    c=np.empty((n,1))
    d=np.empty((n,1))
    k=0
    while k<=n-1:
        if f(a)*f(a+h)<=0:
            c[k]=a
            d[k]=a+h
            k=k+1
        a=a+h
    return c,d    

def newton(fname,dfname,x0,tol,N,m):
# 功能：牛顿迭代法及其修改形式
# 输入：初值x0,最大迭代步数N，误差限tol，m=1为牛顿迭代法，m>1为修改的牛顿迭代法
# 输出：近似根y，迭代次数k    
    y=x0
    x0=y+2*tol
    k=0
    while np.abs(x0-y)>tol and k<N:
        k=k+1
        x0=y
        y=x0-m*fname(x0)/dfname(x0)
        
    if k==N:
        print("warning")
        
    return y,k

def eigen_value_solution(eta,L,var,Num_Root):
# 功能：求解一维特征值 
# 输入：eta:相关长度   L：区域长度   var:方差   Num_Root: 前N个实根
# 输出：Lamda：特征值  w0：对应特征方程正实根   
    w0=np.empty((Num_Root,1))
    lamda=np.empty((Num_Root,1))
    cumulate_lamda=np.empty((Num_Root,1))
    ##############################################################################
    ##定义方程形式
    #########################################################################
    def ff(x):
        ff=(eta**2*x**2-1)*np.sin(x*L)-2*eta*x*np.cos(x*L)
        return ff    
    ##############################################################################
    ##定义方程导数形式
    #########################################################################
    def dff(x):
        dff=(2*eta**2*x-1)*np.sin(x*L)+(eta**2*x**2-1)*np.cos(x*L)*L-2*eta*np.cos(x*L)+2*eta*x*np.sin(x*L)*L
        return dff 
    ##用函数搜索隔根区间
    c,d=search(ff,0.00001,0.00001,Num_Root)
    w00=(c+d)/2    
    #%用牛顿法精确求解
    for i in range(Num_Root):
       w0[i],k= newton(ff,dff,w00[i],1e-8,10000,1)
    ## 根据特征方程正实根，计算特征值λ（Lamda） %%%%
    for flag in range(Num_Root):
        lamda[flag]=2*eta*var/(eta**2*w0[flag]**2+1)
        if flag==0:
            cumulate_lamda[flag]=lamda[flag]
        else:
            cumulate_lamda[flag]=lamda[flag]+cumulate_lamda[flag-1]       
    return lamda,w0,cumulate_lamda

def sort_lamda(lamda_x,w0_x,lamda_y,w0_y,domain,var,weight):
#  功能：二维特征值组合并排序
#  输入：Lamda_x，w0_x:x方向特征值以及对应特征方程实根，Lamda_y,w0_y:y方向特征值以及对应特征方程实根
#       Domain：矩形域范围，var：方差
#       N_X,N_Y:X,Y方向特征值个数
#  输出：lamda:按递减顺序排列的二维特征值
#       w_x,w_y:特征值对应特征方程在不同方向上的正实根
#       n：特征值截断个数，权重weight, cum_lamda:特征根累计值
    n_x=len(w0_x)
    n_y=len(w0_y)
    num=n_x*n_y
    lamda_2d=np.zeros((num,1))
    flag=0
    lamda_index=list()
    for i in range(n_x):
        for j in range(n_y):
            lamda_2d[flag]=lamda_x[i]*lamda_y[j]/var
            lam_ind=[lamda_2d[flag],i,j]          
            lamda_index.append(lam_ind)
            flag=flag+1    
    lamda_index_sorted=sorted(lamda_index, key = lambda x: x[0], reverse=True)    
    sum_lamda=np.zeros((num,1))
    lamda_all=np.zeros((num,1))
    w_x_all=np.zeros((num,1))
    w_y_all=np.zeros((num,1))
    # sum_lamda[0]=lamda_index_sorted[0][0]    
    lab=1    
    for k in range(num):
        lamda_all[k]=lamda_index_sorted[k][0]
        w_x_all[k]=w0_x[lamda_index_sorted[k][1]]
        w_y_all[k]=w0_y[lamda_index_sorted[k][2]]        
        if k==0:
            sum_lamda[k]=lamda_index_sorted[k][0]
        else:
            sum_lamda[k]=sum_lamda[k-1]+lamda_index_sorted[k][0]           
        if lab and sum_lamda[k]/domain/var>=weight:
            n=k+1
            lab=0       
    fig, ax1 = plt.subplots()
    ax1.plot(range(num),lamda_all/domain/var)    
    ax1.set_xlabel('n')
    ax1.set_ylabel('Lamda 2D / (Domain*Var)')  
    plt.title('Series of Engenvalues in 2 Demensions')    
    
    fig, ax2 = plt.subplots()
    ax2.plot(range(num),sum_lamda/domain/var)    
    ax2.set_xlabel('n')
    ax2.set_ylabel('cumulate Lamda/ (Domain*Var)')  
    plt.title('Finite Sums')  
    cum_lamda=np.zeros((n,1))
    lamda=np.zeros((n,1))
    w_x=np.zeros((n,1))
    w_y=np.zeros((n,1))        
    for kk in range(n):
        lamda[kk]=lamda_all[kk]
        w_x[kk]=w_x_all[kk]
        w_y[kk]=w_y_all[kk]
        cum_lamda[kk]=sum_lamda[kk] 
    return lamda,w_x,w_y,n,cum_lamda
  
def eigen_func(n,w,eta,L,x):
#功能：计算特征值对应的特征函数值     
#输入：n:特征值截断个数  w：特征方程正实根  eta:相关长度    L:区域长度   x:位置
#输出：f:特征值对应的特征函数值
    f=np.empty((n,1))
    for i in range(n):
        f[i]=(eta*w[i]*np.cos(w[i]*x)+np.sin(w[i]*x))/np.sqrt((eta**2*w[i]**2+1)*L/2+eta)
    return f

def k2(kesi,f_x,f_y,mean_logk,lamda):
#渗透率计算函数  
    logk=mean_logk+torch.sum(torch.sqrt(lamda)*kesi*f_x*f_y,1)
    kk=torch.exp(logk)
    return kk

def eigen_func2(n,w,eta,L,x):
#计算特征值对应的特征函数值     
#输入：   n:特征值截断个数  w：特征方程正实根  eta:相关长度    L:区域长度   x:位置
#输出：   f:特征值对应的特征函数值
    f=torch.empty((n,len(x)))
    for i in range(n):
        f[i,:]=(eta*w[i]*torch.cos(w[i]*x)+torch.sin(w[i]*x))/torch.sqrt((eta**2*w[i]**2+1)*L/2+eta)
    return f

#########################################################
#数据准备
#########################################################
#渗透率场设置
mean_logk=0
var1=1.0
L_x= 1020    #区域长度
L_y= 1020
eta=408   #相关长度
nx=config.nx     #网格个数
ny=config.ny
nt=config.nt      #时间步数
L_t=config.L_t     #时间总长
domain=L_x*L_y
weight=0.8
Ss=0.0001
x=np.arange(1,52,1)
x=x*20
y=np.arange(1,52,1)
y=y*20
t=np.linspace(0.2,10,50)
h_boun1_0=202
h_boun2_0=200
#设备设置
device = config.device
#Data Processing
bigx=100
L_x=L_x/bigx
L_y= L_y/bigx
eta=eta/bigx
domain=L_x*L_y
x=x/bigx
y=y/bigx
Ss=Ss*bigx*bigx
dx=20/bigx
dy=20/bigx
dt=0.2
#h的无量纲化
h_boun1=(h_boun1_0-h_boun2_0)/(h_boun1_0-h_boun2_0)
h_boun2=(h_boun2_0-h_boun2_0)/(h_boun1_0-h_boun2_0)

#########################################################
#生成随机数组，生成渗透率场实现
#########################################################
#计算所需特征值个数
n_eigen=50
lamda_x,w_x0,cumulate_lamda_x=eigen_value_solution(eta,L_x,var1,n_eigen)
fig, ax1 = plt.subplots()
ax1.plot(range(1,n_eigen+1),lamda_x/L_x,label='eta=408, lamda_x/L')
lamda_y,w_y0,cumulate_lamda_y=eigen_value_solution(eta,L_y,var1,n_eigen)
ax1.plot(range(1,n_eigen+1),lamda_y/L_y,label='eta=408, lamda_y/L')
ax1.set_xlim([1,n_eigen])
plt.legend()
ax1.set_xlabel('n')
ax1.set_ylabel('lamda/L')
fig, ax = plt.subplots()
ax.plot(range(1,n_eigen+1),cumulate_lamda_x/L_x,label='eta=408, sum_lamda_x/L')
ax.plot(range(1,n_eigen+1),cumulate_lamda_y/L_y,label='eta=408, sum_lamda_y/L')
ax.set_xlim([1,n_eigen])
plt.legend()
ax.set_xlabel('n')
ax.set_ylabel('sum_lamda/L')
#二维特征值计算，混合，排序，截断
lamda_xy,w_x,w_y,n,cum_lamda=sort_lamda(lamda_x,w_x0,lamda_y,w_y0,domain,var1,weight)
#根据weight获取所需计算特征值个数,并计算特征值以及特征函数值
n_eigen=n
fn_x=[]
fn_y=[]
for i_x in range(nx):
    f_x=eigen_func(n_eigen,w_x,eta,L_x,x[i_x])
    fn_x.append([f_x,x[i_x]])  
for i_y in range(ny):
    f_y=eigen_func(n_eigen,w_y,eta,L_y,y[i_y])
    fn_y.append([f_y,y[i_y]])
#生成随机数组，生成渗透率场实现
seed_n=38
np.random.seed(seed_n)
n_logk=1  #渗透率场实现个数
num_epoch=config.num_epoch_hyper
N_h=config.N_h
kesi=np.zeros((n_logk,n_eigen))   #随机数数组
logk=np.empty((nx,ny))       #渗透率场数组
for i_logk in range(n_logk):
    kesi[i_logk,:]=np.random.randn(n_eigen)   #随机数数组 
    #由随机数计算渗透率场
    for i_x in range(nx):
        for i_y in range(ny):
            logk[i_y,i_x]=mean_logk
    #            logk[i_logk,i_x,i_y]=logk[i_logk,i_x,i_y]+np.sum(np.sqrt(lamda_xy)*f_x*f_y*kesi[i_logk,:])
            for i_eigen in range(n_eigen):
                logk[i_y,i_x]=logk[i_y,i_x]+np.sqrt(lamda_xy[i_eigen])*fn_x[i_x][0][i_eigen]*fn_y[i_y][0][i_eigen]*kesi[i_logk,i_eigen]
#渗透率场对数转化
k=np.exp(logk)
#数据空间定义
hh=np.empty((nt,nx,ny))

#########################################################
#渗透率场写入程序并进行模拟
#########################################################
#修改工作目录
path = "../"
# 查看当前工作目录
retval = os.getcwd()
print ("当前工作目录为 %s" % retval)
# 修改当前工作目录
os.chdir( path )
# 查看修改后的工作目录
retval = os.getcwd()
print ("目录修改成功 %s" % retval)
for i_k in range(n_logk):
    #渗透率数据写入    
    #1 数组转化为字符串    '
    str_line_all=[]
    for i_x in range(nx):
        str_line='       '
        for i_y in range(ny):
            if k[i_x,i_y]<10:
                str_k='%.7f'%k[i_x,i_y]
            else:
                str_k='%.6f'%k[i_x,i_y]
            #str_k='%.7f'%k[i_k,i_x,i_y]
            str_line=str_line+str_k
            if i_y<ny-1:
                str_line=str_line+'       '
        str_line_all.append(str_line)          
    #2 渗透率字符串数据写入 
    f_write=open("transmissivity_new.dat",'w')   
    for i_x in range(nx):
        f_write.write(str_line_all[i_x]+'\n')
    f_write.close()    
    #3 重命名渗透率文件
    old_name='transmissivity.dat'
    old=str(i_k)
    new_name='transmissivity'+old+'.dat'
    os.rename(old_name,new_name)    
    old_name='transmissivity_new.dat'
    new_name='transmissivity.dat'
    os.rename(old_name,new_name) 
    #运行modflow
    os.system('mf2005 singlephase.MFN')  
    #读取观测结果
    f_read = open("final_head.dat")  
    new_observation_name='observation_k='+str(i_k+1)+'.dat'
    f_write=open(new_observation_name,'w')          
    line = f_read.readline() 
    label0=line[-21:-1]         
    while line:
    #print (line) 
        l_end=line[-21:-1]
    #print(l_end)
        if operator.ne(l_end,label0):
            f_write.write(line)
        line = f_read.readline()
    f_read.close()
    f_write.close()
    #观测结果写入矩阵
    f_read = open(new_observation_name)          
    line = f_read.readline() 
    num_in_line=re.findall(r'\-?\d+\.?\d*', line)
    #num_in_line=re.findall(r'\d+\.?\d*', line)
    n_x=len(num_in_line)
    ii=0       
    while line:
    #print (line)
        for i_x in range(n_x):
            line_array=np.array(num_in_line).astype(float)
            hh[ii,i_x,:]=line_array
            line = f_read.readline()
            num_in_line=re.findall(r'\-?\d+\.?\d*', line)
        ii=ii+1
    f_read.close()
    #删除文件transmissivity, observation
    os.remove('transmissivity'+old+'.dat')
    os.remove( new_observation_name)


# 添加噪声
hh_origion=hh.copy()

h_dif=np.max(hh,0)-np.min(hh,0)

if config.use_noise==True:
    noise = config.noise      
    hh = hh + noise*h_dif*np.random.uniform(-1,1,hh.shape)
t_train=18
hh=np.vstack((hh[0:t_train,:,:],hh_origion[t_train:nt,:,:]))#只有前18步才有扰动，所以把扰动后的前18步和扰动前的后32步拼接

# plt.show()
