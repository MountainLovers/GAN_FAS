import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--epoch', type=int, default=30, help='epoch')
parser.add_argument('--gpu_ids', type=str, default='0,1,2', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')


parser.add_argument('--train_file_list', type=str, default='/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols/Protocol_2/Train_32_proto.txt', help='train_file_list')
parser.add_argument('--dev_file_list', type=str, default='/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols/Protocol_2/Dev_32_proto.txt', help='dev_file_list')
parser.add_argument('--test_file_list', type=str, default='/mnt/hdd.user/datasets/FAS/Oulu-NPU/Protocols/Protocol_2/Test_32_proto.txt', help='test_file_list')

parser.add_argument('--name', type=str, default='tmp',
                    help='name of the experiment. It decides where to store samples and models')
                    
parser.add_argument('--model', type=str, default='model3', help='model in ablation experiment,model1 model2 model3')

parser.add_argument('--data_balance', dest="data_balance", action='store_true', help='data_balance, it is goo for oulu p3/4')
parser.add_argument('--w_cls', type=int, default=1, help='weight of cls loss')
parser.add_argument('--w_NP', type=float, default=5, help='weight of NP loss')
parser.add_argument('--w_L1', type=float, default=1, help='weight of L1 loss')
parser.add_argument('--w_gan', type=int, default=1, help='weight of gan loss')

# training parameters
parser.add_argument('--lr_D', type=float, default=3e-5, help='initial discriminator learning rate for adam')
parser.add_argument('--lr_G', type=float, default=3e-5, help='initial generator learning rate for adam')
parser.add_argument('--lr_C', type=float, default=3e-5, help='initial classifier learning rate for adam')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

# debug parameters
parser.add_argument('--seed', type=int, default=-1, help='random seed')
parser.add_argument('--debug', type=int, default=1, help='debug model. save pics and debug outputs')

# parse opt
import os
from util import util
import torch
opt = parser.parse_args()
expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
util.mkdirs(expr_dir)
str_ids = opt.gpu_ids.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])
