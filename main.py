from dataset import AlignedDataset
from torch.utils.data import DataLoader
from torch import nn
from model import FaceModel
from options import opt
import torchvision.utils as vutils
import os
import sys
import torch
from statistics import PADMeter
import logging
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import  WeightedRandomSampler
from test import eval_model
import matplotlib.pyplot as plt
import numpy as np
import random

file_name = os.path.join(opt.checkpoints_dir, opt.name,"log")
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG,
                    filename= file_name,filemode="w")
run_dir = os.path.join(opt.checkpoints_dir, opt.name,"runs")
if not os.path.exists(run_dir):
    os.makedirs(run_dir)
writer = SummaryWriter(log_dir=run_dir)

logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("%s/%s/debug_%s_{time}.log"%(opt.checkpoints_dir, opt.name, opt.name), rotation="500 MB", level="TRACE")
logger.add("%s/%s/info_%s_{time}.log"%(opt.checkpoints_dir, opt.name, opt.name), rotation="500 MB", level="INFO")

MAX_SIG_ONE_PIC = 8
SIG_SCALE=30

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


def save_pic(ret, k, save_dir, e):
    total = ret['fake_B'].shape[0]
    st = 0
    cnt = 0
    while st < total:
        cnt += 1
        plt.figure(dpi=300)
        for ii in range(k):
            if st + ii >= total:
                break
            plt.subplot(k, 1, ii+1)
            plt.ylim(-SIG_SCALE,SIG_SCALE)
            plt.plot(ret['fake_B'][st+ii])
        
        for ii in range(k):
            if st + ii >= total:
                break
            plt.subplot(k, 1, ii+1)
            plt.ylim(-SIG_SCALE,SIG_SCALE)
            plt.plot(ret['real_B'][st+ii])

        plt.legend(labels=["fake","real"],loc="lower right",fontsize=6)
        plt.savefig("%s/epoch_%d_%d.png" % (save_dir, e, cnt))
        plt.close()
        
        st += k

if __name__ == '__main__':
    # 设置随机数种子
    seed = opt.seed
    d_flag = opt.debug
    if seed != -1:
        setup_seed(seed)
    else:
        torch.backends.cudnn.benchmark = True
    best_res = 101

    img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "res")
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    train_batch_size = opt.batch_size
    test_batch_size = opt.batch_size
    
    train_file_list = opt.train_file_list
    dev_file_list = opt.dev_file_list
    test_file_list = opt.test_file_list
    model = FaceModel(opt,isTrain = True,input_nc = 3)

    loss_names = model.loss_names
    val_loss_names = model.val_loss_names

    test_data_loader = DataLoader(AlignedDataset(test_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=4,drop_last=True)
    dev_data_loader = DataLoader(AlignedDataset(dev_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=4,drop_last=True)

    train_dataset = AlignedDataset(train_file_list) 
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle = True,num_workers=4,drop_last=True)

    writer.iter = 0
    for e in range(opt.epoch):
        logger.info("EPOCH {}".format(e))
        logger.trace("model.train() start")
        model.train()
        # print("!!!!!!!!!! model.train() ok !!!!!!!!!!!!")
        logger.trace("model.train() ok")
        pad_meter_train = PADMeter()
        # print("!!!!!!!!!! pad_meter_train ok !!!!!!!!!!!!")
        # logger.trace("pad_meter_train ok")
        itern = len(train_data_loader)
        for i, data in enumerate(train_data_loader):
            # print("!!!!!!!!!!!! BATCH {} !!!!!!!!!!!!!".format(i))
            logger.trace("Train Iter {}".format(i))
            model.set_input(data)
            model.optimize_parameters(i)
            # print("output: {}".format(model.output))
            class_output = nn.functional.softmax(model.output, dim=1)
            # print("label: {}".format(model.label.cpu().data.numpy()))
            # print("class_output: {}".format(class_output.cpu().data.numpy()))
            pad_meter_train.update(model.label.cpu().data.numpy(),
                             class_output.cpu().data.numpy())

            # writer.add_scalars('train_loss/C', {'C': model.get_current_losses()['C']}, i+ len(train_data_loader) *e)
            # writer.add_scalars('train_loss/G', {'G_GAN_loss': model.get_current_losses()['G_GAN']}, i+ len(train_data_loader) *e)
            # writer.add_scalars('train_loss/G', {'G_MSE': model.get_current_losses()['G_MSE']}, i+ len(train_data_loader) *e)
            # writer.add_scalars('train_loss/D', {'D_real_loss': model.get_current_losses()['D_real']}, i+ len(train_data_loader) *e)
            # writer.add_scalars('train_loss/D', {'D_fake_loss': model.get_current_losses()['D_fake']}, i+ len(train_data_loader) *e)
            # writer.add_scalars('train_loss/D', {'D': model.get_current_losses()['D']}, i+ len(train_data_loader) *e)
            # writer.add_scalars('train_loss/G', {'G': model.get_current_losses()['G']}, i+ len(train_data_loader) *e)
            for name in loss_names:
                writer.add_scalars("train_loss/"+name[0], {name: model.get_current_losses()[name]}, i+ len(train_data_loader) *e)
            

            train_d_flag = d_flag if d_flag > 0 else 20
            if i % train_d_flag == 0:
                # logging.info(model.get_current_losses())
                # logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                #     pad_meter=pad_meter_train))
                #############################
                #######   show loss  ########
                #############################
                logger.info("Epoch [{}/{}] - TRAIN iter [{}/{}]: loss: {}".format(e, opt.epoch, i, itern, model.get_current_losses()))


                #############################
                ####    show padmeter  ######
                #############################
                pad_meter_train.get_eer_and_thr()
                pad_meter_train.get_hter_apcer_etal_at_thr(pad_meter_train.threshold)
                pad_meter_train.get_accuracy(pad_meter_train.threshold)
                tn, fn, tp, fp = pad_meter_train.get_four(pad_meter_train.threshold)
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: TN {} FN {} TP {} FP {}'.format(
                    e, opt.epoch, i, itern, tn, fn, tp, fp))
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    e, opt.epoch, i, itern, pad_meter=pad_meter_train))
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
                    e, opt.epoch, i, itern, pad_meter=pad_meter_train))
                writer.add_scalars('train_padmeter/ErrorRate', {'APCER': pad_meter_train.apcer}, i+ len(train_data_loader) *e)
                writer.add_scalars('train_padmeter/ErrorRate', {'BPCER': pad_meter_train.bpcer}, i+ len(train_data_loader) *e)
                writer.add_scalars('train_padmeter/ErrorRate', {'ACER': pad_meter_train.acer}, i+ len(train_data_loader) *e)
                writer.add_scalars('train_padmeter/AUC', {'AUC': pad_meter_train.auc}, i+ len(train_data_loader) *e)
                
            # save the lastest train pic
            #############################
            ####    save pic    #########
            #############################
            train_img_save_dir = os.path.join(img_save_dir, "train")
            if not os.path.exists(train_img_save_dir):
                os.makedirs(train_img_save_dir)
            if d_flag > 0:
                if i % d_flag == 0:
                    ret = model.get_current_sigs()
                    save_pic(ret, MAX_SIG_ONE_PIC, train_img_save_dir, e)
                    logger.trace("Epoch [{}] - TRAIN iter [{}/{}]: save figs ok".format(e, i, itern))
            elif d_flag == -1:
                if i == itern - 1:
                    ret = model.get_current_sigs()
                    save_pic(ret, MAX_SIG_ONE_PIC, train_img_save_dir, e)
                    logger.trace("Epoch [{}] - TRAIN iter [{}/{}]: save figs ok".format(e, i, itern))


        if e%1==0:
            model.eval()
            pad_dev_mater, _, val_losses = eval_model(dev_data_loader,model)
            for ii, vl in enumerate(val_losses):
                
                # writer.add_scalars('val_loss/C', {'C': vl['C']}, ii+ len(dev_data_loader) *e)
                # writer.add_scalars('val_loss/G', {'G_GAN_loss': vl['G_GAN']}, ii+ len(dev_data_loader) *e)
                # writer.add_scalars('val_loss/G', {'G_MSE': vl['G_MSE']}, ii+ len(dev_data_loader) *e)
                # writer.add_scalars('val_loss/D', {'D_real_loss': vl['D_real']}, ii+ len(dev_data_loader) *e)
                # writer.add_scalars('val_loss/D', {'D_fake_loss': vl['D_fake']}, ii+ len(dev_data_loader) *e)
                # writer.add_scalars('val_loss/D', {'D': vl['D']}, ii+ len(dev_data_loader) *e)
                # writer.add_scalars('val_loss/G', {'G': vl['G']}, ii+ len(dev_data_loader) *e)
                for name in val_loss_names:
                    writer.add_scalars("val_loss/"+name[0], {name: vl[name]}, ii+ len(dev_data_loader) *e)
                

            pad_meter, sigs, test_losses = eval_model(test_data_loader,model)
            for ii, vl in enumerate(test_losses):
                # writer.add_scalars('test_loss/C', {'C': vl['C']}, ii+ len(test_data_loader) *e)
                # writer.add_scalars('test_loss/G', {'G_GAN_loss': vl['G_GAN']}, ii+ len(test_data_loader) *e)
                # writer.add_scalars('test_loss/G', {'G_MSE': vl['G_MSE']}, ii+ len(test_data_loader) *e)
                # writer.add_scalars('test_loss/D', {'D_real_loss': vl['D_real']}, ii+ len(test_data_loader) *e)
                # writer.add_scalars('test_loss/D', {'D_fake_loss': vl['D_fake']}, ii+ len(test_data_loader) *e)
                # writer.add_scalars('test_loss/D', {'D': vl['D']}, ii+ len(test_data_loader) *e)
                # writer.add_scalars('test_loss/G', {'G': vl['G']}, ii+ len(test_data_loader) *e)
                for name in val_loss_names:
                    writer.add_scalars("test_loss/"+name[0], {name: vl[name]}, ii+ len(test_data_loader) *e)

            pad_meter.get_eer_and_thr()
            pad_dev_mater.get_eer_and_thr()

            pad_meter.get_hter_apcer_etal_at_thr(pad_dev_mater.threshold)
            pad_meter.get_accuracy(pad_dev_mater.threshold)
            # logging.info("epoch %d test"%e)
            # logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                # pad_meter=pad_meter))
            tn, fn, tp, fp = pad_meter.get_four(pad_dev_mater.threshold)
            logger.info('Epoch [{}/{}] - TEST: TN {} FN {} TP {} FP {}'.format(
                e, opt.epoch, tn, fn, tp, fp))
            logger.info('Epoch [{}/{}] - TEST: HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            logger.info('Epoch [{}/{}] - TEST: APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            
            writer.add_scalars('test_padmeter/ErrorRate', {'APCER': pad_meter.apcer}, e)
            writer.add_scalars('test_padmeter/ErrorRate', {'BPCER': pad_meter.bpcer}, e)
            writer.add_scalars('test_padmeter/ErrorRate', {'ACER': pad_meter.acer}, e)
            writer.add_scalars('test_padmeter/AUC', {'AUC': pad_meter.auc}, e)
            
            test_img_save_dir = os.path.join(img_save_dir, "test")
            if not os.path.exists(test_img_save_dir):
                os.makedirs(test_img_save_dir)
            if d_flag > 0:
                for i_sigs, ret in enumerate(sigs):
                    if i_sigs % d_flag != 0:
                        continue
                    save_pic(ret, MAX_SIG_ONE_PIC, test_img_save_dir, e)
            elif d_flag == -1:
                ret = sigs[-1]
                save_pic(ret, MAX_SIG_ONE_PIC, test_img_save_dir, e)
            
            logger.trace("Epoch [{}] - TEST: save figs ok".format(e))

            is_best = pad_meter.hter <= best_res
            best_res = min(pad_meter.hter, best_res)
            if is_best:
                best_name = "best"
                model.save_networks(best_name)

        filename = "lastest"
        model.save_networks(filename)