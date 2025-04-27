import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from scipy.io import savemat
from torch import optim
from torch.autograd import Variable
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from calflops import calculate_flops
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import time
import os
import math
from tqdm import tqdm
import logging
from models.DBMGNet import DBMGNet
logging.basicConfig(filename='logs/DBMGNet.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')    
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston', 'LK'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=20, help='number of seed') 
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=20, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='the side length of a single patch')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate') #5e-4
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=1.1e-3, help='weight_decay') 
parser.add_argument('--dual', type=bool, default=False, help='dual')
parser.add_argument('--load', type=str, default='exp/DBMGNet/Indian/bs64_lr0.0001_epoch200_testF20_patch7.pth', help='load')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
#-------------------------------------------------------------------------------


def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1-(epoch) / max_epoch, power), 8)

def warm_up_learning_rate_adjust(init_lr, epoch, warm_epoch, max_epoch, optimizer):
    for param_group in optimizer.param_groups:
        if epoch < warm_epoch:
            param_group['lr'] = init_lr*(1-math.cos(math.pi/2*(epoch+1)/(warm_epoch)))
        else:
            param_group['lr'] = init_lr*(math.cos(math.pi*(epoch-warm_epoch)/max_epoch)+1)/2

def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------

def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)

    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize

    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]

    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]

    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]

    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)

    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------

def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------

def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()
#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        optimizer.zero_grad()

        batch_pred, out1, out2 = model(batch_data)
        loss = criterion(batch_pred, batch_target) + criterion(out1, batch_target) + criterion(out2, batch_target) + kl_divergence_loss(out1, out2) + kl_divergence_loss(out2, out1) 

        loss.backward()
        optimizer.step()       

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre
#-------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred, out1, out2 = model(batch_data)
        loss = criterion(batch_pred, batch_target) + criterion(out1, batch_target) + criterion(out2, batch_target) + kl_divergence_loss(out1, out2) + kl_divergence_loss(out2, out1) #+ 1 + cosine_similarity_loss(x_t, x_c)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre

def test_epoch(model, test_loader, criterion, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()   

        batch_pred, _, _ = model(batch_data)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
        tar = np.append(tar, batch_target.data.cpu().numpy())

    return pre 


#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
import torch.nn.functional as F

def kl_divergence_loss(out1, out2):

    out1_log = F.log_softmax(out1, dim=-1)
    out2_soft = F.softmax(out2, dim=-1)
    return F.kl_div(out1_log, out2_soft, reduction='batchmean')
     
#-------------------------------------------------------------------------------

# Parameter Setting
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False
# prepare data
if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
elif args.dataset == 'Trento':
    data = loadmat('./data/Trento.mat')  
elif args.dataset == 'LK':
    data = loadmat('./data/LongKou.mat') 
elif args.dataset == 'Aug':
    data = loadmat('./data/Augsburg_80.mat') 
elif args.dataset == 'Bot':
    data = loadmat('./data/Botswana_20.mat') 

else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')

TR = data['TR']
TE = data['TE']
input = data['input'] #(145,145,200)
label = TR + TE

num_classes = int(np.max(TR))

color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]] #(17,3)
# normalize data by band norm
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)
# data size
height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
#-------------------------------------------------------------------------------
# obtain train and test data
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
#-------------------------------------------------------------------------------
# load data
x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor) #[695, 200, 7, 7]
y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
Label_train=Data.TensorDataset(x_train,y_train)
x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor) # [9671, 200, 7, 7]
y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
Label_test=Data.TensorDataset(x_test,y_test)
x_true=torch.from_numpy(x_true_band.transpose(0,2,1)).type(torch.FloatTensor)
y_true=torch.from_numpy(y_true).type(torch.LongTensor)
Label_true=Data.TensorDataset(x_true,y_true)

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

#-------------------------------------------------------------------------------
# create model
 
ts = {'Indian':200,
      'Pavia':103,
      'Houston':144,
      'Trento':63,
      'LK':270,
      'Aug':180,
      'Bot':145}

nc = {'Indian':16,
      'Pavia':9,
      'Houston':15,
      'Trento':6,
      'LK':9,
      'Aug':7,
      'Bot':14}
 
model = DBMGNet(channels=ts[args.dataset], num_classes=nc[args.dataset], image_size=args.patches, num_layers=1)

model = model.cuda()

# criterion
criterion = nn.CrossEntropyLoss().cuda()
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
os.makedirs(f'exp/DBMGNet/{args.dataset}',exist_ok=True)
weight_root = f'exp/DBMGNet/{args.dataset}'
#-------------------------------------------------------------------------------
if args.flag_test == 'test':
    model.load_state_dict(torch.load(args.load),strict=True)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_parameters count: {str(n_parameters/1000**2)}M")
    logger.info(f"n_parameters count: {str(n_parameters/1000**2)}M")
    print('----------------------------------------------------------')
    
    # Create dummy input tensor for FLOPs calculation
    if args.dataset ==  'Indian':
        input_size = (1, 200, 49)  
    elif args.dataset ==  'Pavia':
        input_size = (1, 103, 49) 
    elif args.dataset ==  'Houston':
        input_size = (1, 144, 49) 
    elif args.dataset ==  'LK':
        input_size = (1, 270, 49) 
    dummy_input = torch.randn(input_size).cuda()
    
    # Calculate FLOPs and convert to GFLOPs
    flops = FlopCountAnalysis(model, dummy_input)
    gflops = flops.total() / 1e9  # Convert to GigaFLOPs
    # Print FLOPs and parameter count
    print(f"GFLOPs: {gflops:.3f} GFLOPs")  # Display GFLOPs to three decimal places
    logger.info(f"GFLOPs: {gflops:.3f} GFLOPs") 
    # print(f"Parameter count: {parameter_count_table(model)}")

    flops, macs1, para = calculate_flops(model, input_shape=input_size, )
    logger.info("para:{}\n,flops:{}".format(para, flops))
#------------------------------------------------------------
    
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    print(" val_OA: {:.4f} val_AA: {:.4f} val_Kappa: {:.4f} ".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    
    color_costom_matrix = np.ones([16, 3])
    if args.dataset == 'Indian':
        dpi_large = 600
        color_costom_matrix = np.array([[79, 170, 72],
                                        [136, 186, 67],
                                        [62, 131, 91],
                                        [54, 132, 68],
                                        [144, 81, 54],
                                        [102, 188, 199],
                                        [255, 255, 255],
                                        [198, 175, 201],
                                        [218, 48, 44],
                                        [120, 34, 35],
                                        [86, 87, 89],
                                        [223, 220, 83],
                                        [217, 142, 52],
                                        [83, 47, 125],
                                        [227, 119, 90],
                                        [157, 86, 151], ])
    elif args.dataset == 'Pavia':
        dpi_large = 600
        color_costom_matrix = np.array([[199, 200, 202],
                                        [109, 177, 70],
                                        [102, 188, 199],
                                        [55, 123, 66],
                                        [73, 73, 179],  # metal sheets
                                        [149, 82, 49],
                                        [116, 45, 121],
                                        [200, 88, 76],  # brick
                                        [223, 220, 83], ])
    elif args.dataset == 'Trento':
        dpi_large = 600
        color_costom_matrix = np.array([[223, 220, 83],
                                        [218, 48, 44],
                                        [199, 200, 202],
                                        [149, 82, 49],
                                        [109, 177, 70], 
                                        [102, 188, 199], ])
    elif args.dataset == 'Houston':
        dpi_large = 1000
        color_costom_matrix = np.array([[79, 170, 72],
                                        [136, 186, 67],
                                        [62, 131, 91],
                                        [54, 132, 68],
                                        [144, 81, 54],
                                        [102, 188, 199],
                                        [255, 255, 255],
                                        [218, 48, 44],  # commercial
                                        [120, 133, 131],  # road
                                        [120, 34, 35],
                                        [50, 101, 67],
                                        [223, 220, 83],
                                        [198, 175, 201],
                                        [83, 47, 125],
                                        [227, 119, 90], ])
    elif args.dataset == 'LK':
        dpi_large = 1000
        color_costom_matrix = np.array([[79, 170, 72],
                                        [136, 186, 67],
                                        [62, 131, 91],
                                        [54, 132, 68],
                                        [144, 81, 54],
                                        [102, 188, 199],
                                        [223, 220, 83],
                                        [218, 48, 44],  # commercial
                                        [120, 133, 131], ])
    else:
        raise ValueError("Unkknow dataset")
    color_costom_matrix = color_costom_matrix/255
    time_start = time.time()
    
    pre_u = test_epoch(model, label_true_loader, criterion, optimizer)
    # plot_tsne(feature, label)
    # plot_umap(feature, label)
    
    time_end = time.time()
    print(f"推理时间为{time_end-time_start}")
    logger.info(f"推理时间为{time_end-time_start}")
    prediction_matrix = np.zeros((height, width), dtype=float)
    for i in range(total_pos_true.shape[0]):
        prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1
    plt.subplot(1, 1, 1)
    plt.imshow(prediction_matrix, colors.ListedColormap(color_costom_matrix))
    plt.xticks([])
    plt.yticks([])
    os.makedirs('image_vision',exist_ok=True)
    plt.savefig(f"image_vision/{args.dataset}/mamba.png", dpi=dpi_large, bbox_inches='tight')
    exit()
    
elif args.flag_test == 'train':
    print("start training")
    tic = time.time()
    best_score = {'epoch': 0, 
                  'score': [0,0,0]}
    for epoch in tqdm(range(args.epoches)): 
        # scheduler.step()
        # adjust_learning_rate(optimizer, epoch, args.epoches, args.learning_rate)
        warm_up_learning_rate_adjust(args.learning_rate, epoch, 10, args.epoches, optimizer) #

        # train model
        model.train()
        train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer)
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t) 
        tqdm.write("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch+1, train_obj, train_acc))
        

        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):         
            model.eval()
            save = False
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if (OA2+AA_mean2+Kappa2) >= sum(best_score['score']):
                save = True
                best_score['epoch'] =  epoch+1
                best_score['score'] =  [OA2, AA_mean2, Kappa2]
                best_score['AA'] = AA2
                print("Epoch: {:03d} val_OA: {:.4f} val_AA: {:.4f} val_Kappa: {:.4f} ".format(epoch+1, OA2, AA_mean2, Kappa2))
            tqdm.write("Epoch: {:03d} val_OA: {:.4f} val_AA: {:.4f} val_Kappa: {:.4f} ".format(epoch+1, OA2, AA_mean2, Kappa2))
            if save:
                torch.save(model.state_dict(),os.path.join(weight_root,f'bs{args.batch_size}_lr{args.learning_rate}_epoch{args.epoches}_testF{args.test_freq}_patch{args.patches}.pth'))
                
    toc = time.time()
    print("Running Time: {:.2f}".format(toc-tic))
    logger.info("Running Time: {:.2f}".format(toc-tic))
    print("**************************************************")

logger.info("##############start##########")
logger.info("Final result:")
logger.info("BestEpoch: {:03d} OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(best_score['epoch'],best_score['score'][0], best_score['score'][1], best_score['score'][2]))
logger.info(f"All_Accuracy:\n {best_score['AA']}")
logger.info("**************************************************")
logger.info("Parameter:")


def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))
        logger.info("{0}: {1}".format(k,v))
print_args(vars(args))
print("BestEpoch: {:03d} OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(best_score['epoch'],best_score['score'][0], best_score['score'][1], best_score['score'][2]))
print(f"All_Accuracy:\n {best_score['AA']}")
logger.info("##############finish##########")
logger.info("\n\n\n")










