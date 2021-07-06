import torch

model_ID = 2 # 0 HCP, 1 TgNN, 2 ANN

if model_ID == 0: # HCP
    use_HCP = True
    use_ANN = False
if model_ID == 1: # TgNN
    use_HCP = False
    use_ANN = False
if model_ID == 2: # ANN
    use_HCP = False
    use_ANN = True
    
num_epoch_hyper=2000

Nf_hyper=1000 #2000
N_boun_hyper = 10000
N_no_flow = N_boun_hyper
N_h = 10 # the number of observations at each step
BATCH_SIZE_hyper = int(N_h*18/2)

seed = 38
plot_value_surface = False

nx=51     #网格个数
ny=51
nt=50      #时间步数
L_t=10     #时间总长

use_noise = False
noise = 0.6

use_outlier = False
outlier_per = 0.1

#设备设置
device = torch.device('cuda:1')
#device = torch.device('cpu')

