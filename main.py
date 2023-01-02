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
logger.add("checkpoints/%s/debug_%s_{time}.log"%(opt.name, opt.name), rotation="500 MB", level="TRACE")
logger.add("checkpoints/%s/info_%s_{time}.log"%(opt.name, opt.name), rotation="500 MB", level="INFO")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # 设置随机数种子
    seed = opt.seed
    d_flag = opt.debug
    if seed != -1:
        setup_seed(seed)
    else:
        torch.backends.cudnn.benchmark = True
    best_res = 101
    train_batch_size = opt.batch_size
    test_batch_size = opt.batch_size
    
    train_file_list = opt.train_file_list
    dev_file_list = opt.dev_file_list
    test_file_list = opt.test_file_list
    model = FaceModel(opt,isTrain = True,input_nc = 3)
    test_data_loader = DataLoader(AlignedDataset(test_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=1,drop_last=True)
    dev_data_loader = DataLoader(AlignedDataset(dev_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=1,drop_last=True)

    train_dataset = AlignedDataset(train_file_list) 
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle = True,num_workers=1,drop_last=True)

    writer.iter = 0
    for e in range(opt.epoch):
        logger.info("EPOCH [{}] Train".format(e))
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
            model.optimize_parameters()
            # print("output: {}".format(model.output))
            class_output = nn.functional.softmax(model.output, dim=1)
            # print("label: {}".format(model.label.cpu().data.numpy()))
            # print("class_output: {}".format(class_output.cpu().data.numpy()))
            pad_meter_train.update(model.label.cpu().data.numpy(),
                             class_output.cpu().data.numpy())

            writer.add_scalars('train_loss/C', {'C': model.get_current_losses()['C']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/G', {'G_GAN_loss': model.get_current_losses()['G_GAN']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/G', {'G_NP_loss': model.get_current_losses()['G_NP']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/G', {'G_L1': model.get_current_losses()['G_L1']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/D', {'D_real_loss': model.get_current_losses()['D_real']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/D', {'D_fake_loss': model.get_current_losses()['D_fake']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/D', {'D': model.get_current_losses()['D']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/G', {'G': model.get_current_losses()['G']}, i+ len(train_data_loader) *e)
            if i % 20 ==0:
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
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    e, opt.epoch, i, itern, pad_meter=pad_meter_train))
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
                    e, opt.epoch, i, itern, pad_meter=pad_meter_train))
                writer.add_scalars('train_padmeter/ErrorRate', {'APCER': pad_meter_train.apcer}, i+ len(train_data_loader) *e)
                writer.add_scalars('train_padmeter/ErrorRate', {'BPCER': pad_meter_train.bpcer}, i+ len(train_data_loader) *e)
                writer.add_scalars('train_padmeter/ErrorRate', {'ACER': pad_meter_train.acer}, i+ len(train_data_loader) *e)
                writer.add_scalars('train_padmeter/AUC', {'AUC': pad_meter_train.auc}, i+ len(train_data_loader) *e)

                if d_flag:
                    #############################
                    ####    save pic    #########
                    #############################
                    img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "res")
                    train_img_save_dir = os.path.join(img_save_dir, "train")
                    test_img_save_dir = os.path.join(img_save_dir, "test")
                    if not os.path.exists(img_save_dir):
                        os.makedirs(img_save_dir)
                    if not os.path.exists(train_img_save_dir):
                        os.makedirs(train_img_save_dir)
                    if not os.path.exists(test_img_save_dir):
                        os.makedirs(test_img_save_dir)

                    ret = model.get_current_sigs()

                    # save sigal figure
                    plt.figure(dpi=300)

                    for ii in range(ret['fake_B'].shape[0]):
                        plt.subplot(ret['fake_B'].shape[0], 1, ii+1)
                        plt.plot(ret['fake_B'][ii])
                    # plt.savefig("%s/epoch_%d_fake.png" % (img_save_dir, e))

                    for ii in range(ret['real_B'].shape[0]):
                        plt.subplot(ret['real_B'].shape[0], 1, ii+1)
                        plt.plot(ret['real_B'][ii])

                    plt.legend(labels=["fake","real"],loc="lower right",fontsize=6)
                    plt.savefig("%s/epoch_%d.png" % (train_img_save_dir, e))

                    plt.close()
                    logger.trace("Epoch [{}] - TRAIN iter [{}/{}]: save figs ok".format(e, i, itern))

                    # vutils.save_image(ret['fake_B'], "%s/epoch_%d_fake.png" % (img_save_dir, e), normalize=True)
                    # vutils.save_image(ret['real_B'], "%s/epoch_%d_real.png" % (img_save_dir, e), normalize=True)


        if e%1==0:
            logger.info("Epoch [{}] Test".format(e))
            model.eval()
            pad_dev_mater, _, val_losses = eval_model(dev_data_loader,model)
            for ii, vl in enumerate(val_losses):
                writer.add_scalars('val_loss/C', {'C': vl['C']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/G', {'G_GAN_loss': vl['G_GAN']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/G', {'G_NP_loss': vl['G_NP']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/G', {'G_L1': vl['G_L1']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/D', {'D_real_loss': vl['D_real']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/D', {'D_fake_loss': vl['D_fake']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/D', {'D': vl['D']}, ii+ len(dev_data_loader) *e)
                writer.add_scalars('val_loss/G', {'G': vl['G']}, ii+ len(dev_data_loader) *e)

            pad_meter, sigs, test_losses = eval_model(test_data_loader,model)
            for ii, vl in enumerate(test_losses):
                writer.add_scalars('test_loss/C', {'C': vl['C']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/G', {'G_GAN_loss': vl['G_GAN']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/G', {'G_NP_loss': vl['G_NP']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/G', {'G_L1': vl['G_L1']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/D', {'D_real_loss': vl['D_real']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/D', {'D_fake_loss': vl['D_fake']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/D', {'D': vl['D']}, ii+ len(test_data_loader) *e)
                writer.add_scalars('test_loss/G', {'G': vl['G']}, ii+ len(test_data_loader) *e)

            pad_meter.get_eer_and_thr()
            pad_dev_mater.get_eer_and_thr()

            pad_meter.get_hter_apcer_etal_at_thr(pad_dev_mater.threshold)
            pad_meter.get_accuracy(pad_dev_mater.threshold)
            # logging.info("epoch %d test"%e)
            # logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                # pad_meter=pad_meter))
            logger.info('Epoch [{}/{}] - TEST: HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            logger.info('Epoch [{}/{}] - TEST: APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            
            writer.add_scalars('test_padmeter/ErrorRate', {'APCER': pad_meter.apcer}, e)
            writer.add_scalars('test_padmeter/ErrorRate', {'BPCER': pad_meter.bpcer}, e)
            writer.add_scalars('test_padmeter/ErrorRate', {'ACER': pad_meter.acer}, e)
            writer.add_scalars('test_padmeter/AUC', {'AUC': pad_meter.auc}, e)

            if d_flag:
                for i_sigs, ret in enumerate(sigs):
                    if i_sigs % 10 != 0:
                        continue
                    # ret = model.get_current_sigs()
                    # save sigal figure
                    plt.figure(dpi=300)

                    for ii in range(ret['fake_B'].shape[0]):
                        plt.subplot(ret['fake_B'].shape[0], 1, ii+1)
                        plt.plot(ret['fake_B'][ii])
                    # plt.savefig("%s/epoch_%d_fake.png" % (img_save_dir, e))

                    for ii in range(ret['real_B'].shape[0]):
                        plt.subplot(ret['real_B'].shape[0], 1, ii+1)
                        plt.plot(ret['real_B'][ii])

                    plt.legend(labels=["fake","real"],loc="lower right",fontsize=6)
                    plt.savefig("%s/epoch_%d_%d.png" % (test_img_save_dir, e, i_sigs))

                    plt.close()
                logger.trace("Epoch [{}] - TEST: save figs ok".format(e))

            is_best = pad_meter.hter <= best_res
            best_res = min(pad_meter.hter, best_res)
            if is_best:
                best_name = "best"
                model.save_networks(best_name)
                logger.trace("Epoch [{}] - TEST: save best network ok, best_res = {}".format(e, best_res))

        filename = "lastest"
        model.save_networks(filename)
        logger.trace("Epoch [{}] - TEST: save lastest network ok".format(e))