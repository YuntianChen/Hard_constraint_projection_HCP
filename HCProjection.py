import torch
import numpy as np
import simulation as simu
import configure as config
from torch.autograd import Variable

seed = config.seed
torch.random.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)

h_boun1_0 = simu.h_boun1_0-200
h_boun2_0 = simu.h_boun2_0-200
device = config.device

def buildA_batch(K, x, y, deltaT, deltaX, deltaY, Ss):
# 构建A矩阵，AH=0， 
# 输入为渗透率K，坐标位置xy，网格在xyt上的长度，Specific storage Ss
# 输出为A矩阵，用于后续映射
    batch_size = x.size()[0]
    A = np.zeros((batch_size, 6))
    Kx1 = (2*K[x + 1, y]*K[x, y]) / (K[x + 1, y]+K[x, y])
    Kx2 = (2*K[x - 1, y]*K[x, y]) / (K[x - 1, y]+K[x, y])
    Ky1 = (2*K[x, y + 1]*K[x, y]) / (K[x, y + 1]+K[x, y])
    Ky2 = (2*K[x, y - 1]*K[x, y]) / (K[x, y - 1]+K[x, y])
    A[:, 0] = torch.tensor(Ss / deltaT).repeat(batch_size)
    A[:, 1:2] = -A[:, 0:1] -(Kx1+Kx2)/deltaX**2 -(Ky1+Ky2)/deltaY**2
    A[:, 2:3] = Kx2/deltaX**2
    A[:, 3:4] = Kx1/deltaX**2
    A[:, 4:5] = Ky2/deltaY**2
    A[:, 5:6] = Ky1/deltaY**2
    return  torch.from_numpy(A).type(torch.FloatTensor).to(device).unsqueeze(1)

def projection_batch(A, H):
# 通用性映射函数，
# 输入为A矩阵，H矩阵（模型预测结果）
# 输出为H_矩阵，是映射后的模型预测结果
    I = torch.eye(A.size()[2]).float().to(device).repeat(A.size()[0], 1, 1)
    invs = torch.inverse(torch.bmm(A, A.permute(0, 2, 1)))
    intermediate = torch.bmm(torch.bmm(A.permute(0, 2, 1), invs), A)
    Hreal = H*(h_boun1_0-h_boun2_0)+h_boun2_0
    Hreal_ = torch.bmm((I - intermediate), Hreal)
    H_ = (Hreal_-h_boun2_0)/(h_boun1_0-h_boun2_0)
    AHreal = torch.bmm(A, Hreal).squeeze(2)
    AHreal_ = torch.bmm(A, Hreal_).squeeze(2)
    AH = torch.bmm(A, H).squeeze(2)
    AH_ = torch.bmm(A, H_).squeeze(2)
    #hc_loss = torch.pow(AHreal,2).mean()
    hc_loss = torch.abs(AH)
    hc_loss_ = torch.abs(AH_)
    return H_.squeeze(2), hc_loss, hc_loss_, Hreal_


def generate_HCP_area (T_test_col, X_test_col, Y_test_col):
# 由核心点生成周围配区域的坐标
# 输入为核心点的TXY
# 输出为配区域内6个点的坐标，每个点都是TXY拼接而成的
    T_test_col_minus = T_test_col - 0.2
    X_test_col_minus = X_test_col - 0.2
    X_test_col_plus = X_test_col + 0.2
    Y_test_col_minus = Y_test_col - 0.2
    Y_test_col_plus = Y_test_col +0.2
    TXY_all_1 = np.hstack((T_test_col_minus,X_test_col,Y_test_col)) 
    TXY_all = np.hstack((T_test_col,X_test_col,Y_test_col)) 
    TXY_all_3 = np.hstack((T_test_col,X_test_col_minus,Y_test_col)) 
    TXY_all_4 = np.hstack((T_test_col,X_test_col_plus,Y_test_col)) 
    TXY_all_5 = np.hstack((T_test_col,X_test_col,Y_test_col_minus)) 
    TXY_all_6 = np.hstack((T_test_col,X_test_col,Y_test_col_plus)) 
    TXY_all_1 = Variable(torch.from_numpy(TXY_all_1).type(torch.FloatTensor), requires_grad=True).to(device)
    TXY_all = Variable(torch.from_numpy(TXY_all).type(torch.FloatTensor), requires_grad=True).to(device)
    TXY_all_3 = Variable(torch.from_numpy(TXY_all_3).type(torch.FloatTensor), requires_grad=True).to(device)
    TXY_all_4 = Variable(torch.from_numpy(TXY_all_4).type(torch.FloatTensor), requires_grad=True).to(device)
    TXY_all_5 = Variable(torch.from_numpy(TXY_all_5).type(torch.FloatTensor), requires_grad=True).to(device)
    TXY_all_6 = Variable(torch.from_numpy(TXY_all_6).type(torch.FloatTensor), requires_grad=True).to(device)
    return TXY_all_1, TXY_all, TXY_all_3, TXY_all_4, TXY_all_5, TXY_all_6


def H_set_boundary (h_pred_pinn_1, h_pred_pinn, h_pred_pinn_3, h_pred_pinn_4, h_pred_pinn_5, h_pred_pinn_6):
# 设置边界条件的H，定压边界条件处Hi-1 = 2*Hi - Hi+1； 无流边界条件处理论上虚网格的H任意，此处设为与边界H相同
# 输入：x-1,x+1,y-1,y+1处的h预测值
# 输出：x-1,x+1,y-1,y+1处的h经过边界调整后的预测值。基于这些数据，可以组合生成整个流场中满足边界条件的包含有虚网格的H场
    # 上一时间步核心位置xyt-1处左右边界处定压
    h_previous_fix = h_pred_pinn_1.cpu().data.numpy() 
    h_previous_fix[0:2601] = 0 # 将t=0对应的初始实现全部设为0，后续经过边界，会将初始实现左边界设为h_boun1_0
    for x_index in range (len(h_previous_fix)):        
        if x_index % 51 == 0: #对应x=0.2, 每隔51个点的【开头】就是一个边界点，直接赋值为边界值
            h_previous_fix[x_index] = 1 # 1 对应于边界条件处归一化后的数值
            h_previous_fix[x_index+50] = 0 # 0 对应于hh.min(),即边界条件处
    h_pred_pinn_1_bc = Variable(torch.from_numpy(h_previous_fix).type(torch.FloatTensor), requires_grad=True).to(device)
    # 核心位置xyt处左右边界处定压
    h_center_fix = h_pred_pinn.cpu().data.numpy() 
    for x_index in range (len(h_center_fix)):
        if x_index % 51 == 0: #对应x=0.2, 每隔51个点的【开头】就是一个边界点，直接赋值为边界值
            h_center_fix[x_index] = 1 # 1 对应于边界条件处归一化后的数值
            h_center_fix[x_index+50] = 0 # 0 对应于hh.min(),即边界条件处
    h_pred_pinn_bc = Variable(torch.from_numpy(h_center_fix).type(torch.FloatTensor), requires_grad=True).to(device)
    # 左侧定压 Hi-1 = 2*Hi - Hi+1, i为边界
    h_left_fix = h_pred_pinn_3.cpu().data.numpy() # 所有核心点x-1处的坐标对应的H，其中只有x=0的点是边界点
    for x_index in range (len(h_left_fix)):
        if x_index % 51 == 0: #对应x=0, 每隔51个点的【开头】就是一个边界点左侧的ghost，要取【后方】两个点计算获得边界点的H
            h_left_fix[x_index+1] = 1 # 1 对应于边界条件处归一化后的数值
            h_left_fix[x_index] = 2*h_left_fix[x_index+1] - h_left_fix[x_index+2]
    h_pred_pinn_3_bc = Variable(torch.from_numpy(h_left_fix).type(torch.FloatTensor), requires_grad=True).to(device)
    # 右侧定压 Hi+1 = 2*Hi - Hi-1, i为边界
    h_right_fix = h_pred_pinn_4.cpu().data.numpy() # 所有核心点x-1处的坐标对应的H，其中只有x=50的点是边界点
    for x_index in range (len(h_right_fix)):
        if (x_index+1) % 51 == 0: #对应x=10.4，每隔51个点的【结尾】就是一个边界点右侧的ghost，要取【前方】两个点计算获得边界点的H
            h_right_fix[x_index-1] = 0
            h_right_fix[x_index] = 2*h_right_fix[x_index-1] - h_right_fix[x_index-2]
    h_pred_pinn_4_bc = Variable(torch.from_numpy(h_right_fix).type(torch.FloatTensor), requires_grad=True).to(device)  
    # 下侧无流 Hi-1 = Hi i为边界。实际此时由于ki-1=0，所以Hi-1的取值可以任意,此处设为与Hi处同。
    h_down_fix = h_pred_pinn_5.cpu().data.numpy() # 所有核心点x-1处的坐标对应的H，其中只有x=50的点是边界点
    for y_index in range (len(h_down_fix)):
        if y_index % 51 == 0: #对应x=0.2处
            h_down_fix[y_index] = 1 # 1 对应于边界条件处归一化后的数值 
            h_down_fix[y_index+50] = 0 # 1 对应于边界条件处归一化后的数值       
        if y_index % 2601 == 0: #对应y=0，每隔2601个点就有连续51个边界点，下侧对应于【开头】的51个点，直接等于【后方】51个点即可            
            h_down_fix[y_index:y_index+51] = h_down_fix[y_index+51:y_index+102]
    h_pred_pinn_5_bc = Variable(torch.from_numpy(h_down_fix).type(torch.FloatTensor), requires_grad=True).to(device)  
    # 上侧无流 Hi+1 = Hi i为边界。实际此时由于ki+1=0，所以Hi+1的取值可以任意。
    h_up_fix = h_pred_pinn_6.cpu().data.numpy() # 所有核心点x-1处的坐标对应的H，其中只有x=50的点是边界点
    for y_index in range (len(h_up_fix)):
        if y_index % 51 == 0: #对应x=0.2处
            h_up_fix[y_index] = 1 # 1 对应于边界条件处归一化后的数值 
            h_up_fix[y_index+50] = 0 # 1 对应于边界条件处归一化后的数值     
        if (y_index+51) % 2601 == 0: #对应y=10.4，每隔2601个点就有连续51个边界点，上侧对应于【结尾】的51个点，直接等于【前方】51个点即可
            h_up_fix[y_index:y_index+51] = h_up_fix[y_index-51:y_index]
    h_pred_pinn_6_bc = Variable(torch.from_numpy(h_up_fix).type(torch.FloatTensor), requires_grad=True).to(device)  
    return h_pred_pinn_1_bc, h_pred_pinn_bc, h_pred_pinn_3_bc, h_pred_pinn_4_bc, h_pred_pinn_5_bc, h_pred_pinn_6_bc


def compare_gradient(H, Ht, Hx, Hy):
# 功能：对比一个batch中，配区域内差分获得的梯度与配点处自动微分获得的梯度的差异
# 输入：一个batch中所有点在配区域内的取值H；配点处自动微分获得的Ht，Hx和Hy
# 输出：txy三个方向的batch内两个梯度相对误差绝对值的最大值、中位数、均值
    HCt = (H[:,1:2] - H[:,0:1])/0.2
    HCx = (H[:,3:4] - H[:,2:3])/0.2/2 
    HCy = (H[:,5:6] - H[:,4:5])/0.2/2
    Ht_diff = np.abs(((HCt-Ht)/Ht).cpu().detach().numpy()) 
    Hx_diff = np.abs(((HCx-Hx)/Hx).cpu().detach().numpy()) 
    Hy_diff = np.abs(((HCy-Hy)/Hy).cpu().detach().numpy()) 
    gradient_diff_max = np.array([np.max(Ht_diff), np.max(Hx_diff), np.max(Hy_diff)])
    gradient_diff_mean = np.array([np.mean(Ht_diff), np.mean(Hx_diff), np.mean(Hy_diff)])
    gradient_diff_median = np.array([np.median(Ht_diff), np.median(Hx_diff), np.median(Hy_diff)])
    #print('gradient_diff_max:', gradient_diff_max)
    #print('gradient_diff_mean:', gradient_diff_mean)
    #print('gradient_diff_median:', gradient_diff_median)
    #HCx2 = (H[:,3:4] - H[:,1:2])/0.2
    #HCx3 = (H[:,1:2] - H[:,2:3])/0.2
    #HCx2_diff = np.abs(((HCx2-HCx)/HCx).cpu().detach().numpy()) 
    #HCx3_diff = np.abs(((HCx3-HCx)/HCx).cpu().detach().numpy())
    #print('HCx_diff_mean:', np.array([np.mean(HCx2_diff), np.mean(HCx3_diff)]))
    gradient_diff = np.hstack([gradient_diff_max, gradient_diff_mean, gradient_diff_median])
    diff_t_ID, diff_x_ID, diff_y_ID = 0,0,0
    return gradient_diff, diff_t_ID, diff_x_ID, diff_y_ID



def compare_loss(temp, k, batch_xf2, Ht, Hxx, Hx, Hyy, Hy, K, k_x_, k_y_ ):
# 功能：根据每个迭代步的结果，对比自动微分和差分格式计算的pde中三项（关于txy的导数项）的差异
# 输入：temp是迭代步的结果，也就是H，对应于周围6个点；k是渗透率场；batch_xf2是6个点中中心点的坐标；Ht，Hx，Hy分别是H自动微分的一阶导数，Hxx，Hyy分别是自动微分二阶导，k_x_，k_y_是渗透率的导数
# 输出：diff是差分与自动微分的差，pde表示自动微分，a表示差分结果，012分别对应于txy。
    x = torch.round(batch_xf2[:,1].detach()*100/20-1).long()
    y = torch.round(batch_xf2[:,2].detach()*100/20-1).long()
    k = torch.from_numpy(k).type(torch.DoubleTensor).to(device)
    temp = temp.type(torch.DoubleTensor).to(device)
    # 调和平均k
    Kx1 = ((2*k[x + 1, y]*k[x, y]) / (k[x + 1, y]+k[x, y])).type(torch.DoubleTensor).to(device)
    Kx2 = ((2*k[x - 1, y]*k[x, y]) / (k[x - 1, y]+k[x, y])).type(torch.DoubleTensor).to(device)
    Ky1 = ((2*k[x, y + 1]*k[x, y]) / (k[x, y + 1]+k[x, y])).type(torch.DoubleTensor).to(device)
    Ky2 = ((2*k[x, y - 1]*k[x, y]) / (k[x, y - 1]+k[x, y])).type(torch.DoubleTensor).to(device)
    # hcloss 中各项
    a0 = -5* (temp[:,1] - temp[:,0])
    a1 = (  -(Kx1+Kx2)*temp[:,1] + Kx1*temp[:,3]  +  Kx2*temp[:,2]  )/0.04
    a2 = (  -(Ky1+Ky2)*temp[:,1] + Ky1*temp[:,5]  +  Ky2*temp[:,4]  )/0.04
    hc_loss_3_item = a0+a1+a2
    # pde loss 中各项          
    pde0 = (-1*Ht).type(torch.DoubleTensor).to(device)
    pde1 = (K*Hxx + k_x_*Hx).type(torch.DoubleTensor).to(device)
    pde2 = (K*Hyy + k_y_*Hy).type(torch.DoubleTensor).to(device)
    pde_loss = (pde0+pde1+pde2).type(torch.DoubleTensor).to(device)
    # 统计指标
    diff0 = np.hstack([torch.abs(a0-pde0).mean().detach().cpu().numpy(), torch.abs(a0-pde0).median().detach().cpu().numpy(), torch.abs(a0-pde0).max().detach().cpu().numpy()])
    diff1 = np.hstack([torch.abs(a1-pde1).mean().detach().cpu().numpy(), torch.abs(a1-pde1).median().detach().cpu().numpy(), torch.abs(a1-pde1).max().detach().cpu().numpy()])
    diff2 = np.hstack([torch.abs(a2-pde2).mean().detach().cpu().numpy(), torch.abs(a2-pde2).median().detach().cpu().numpy(), torch.abs(a2-pde1).max().detach().cpu().numpy()])
    diff = np.hstack([torch.abs(hc_loss_3_item - pde_loss).mean().detach().cpu().numpy(), \
                      torch.abs(hc_loss_3_item - pde_loss).median().detach().cpu().numpy(),\
                      torch.abs(hc_loss_3_item - pde_loss).max().detach().cpu().numpy()])
    pde012 = np.hstack([pde0.mean().detach().cpu().numpy(), pde1.mean().detach().cpu().numpy(), pde2.mean().detach().cpu().numpy()])
    a012 = np.hstack([a0.mean().detach().cpu().numpy(), a1.mean().detach().cpu().numpy(), a2.mean().detach().cpu().numpy()])
    return diff0, diff1, diff2, diff, pde012, a012, hc_loss_3_item




