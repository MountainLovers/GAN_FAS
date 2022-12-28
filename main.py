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
        logger.info("EPOCH {}".format(e))
        logger.trace("model.train() start")
        model.train()
        # print("!!!!!!!!!! model.train() ok !!!!!!!!!!!!")
        logger.trace("model.train() ok")
        pad_meter_train = PADMeter()
        # print("!!!!!!!!!! pad_meter_train ok !!!!!!!!!!!!")
        logger.trace("pad_meter_train ok")
        for i, data in enumerate(train_data_loader):
            # print("!!!!!!!!!!!! BATCH {} !!!!!!!!!!!!!".format(i))
            logger.trace("BATCH {}".format(i))
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
            writer.add_scalars('train_loss/D', {'D_real_loss': model.get_current_losses()['D_real']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/D', {'D_fake_loss': model.get_current_losses()['D_fake']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/D', {'D': model.get_current_losses()['D']}, i+ len(train_data_loader) *e)
            writer.add_scalars('train_loss/G', {'G': model.get_current_losses()['G']}, i+ len(train_data_loader) *e)
            if i %100 ==0:
                pad_meter_train.get_eer_and_thr()
                pad_meter_train.get_hter_apcer_etal_at_thr(pad_meter_train.threshold)
                pad_meter_train.get_accuracy(pad_meter_train.threshold)
                ret = model.get_current_sigs()
                img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "res")
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                logging.info(model.get_current_losses())
                logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    pad_meter=pad_meter_train))
                logger.info("Epoch [{}/{}] iter {}: train_loss: {}".format(e, opt.epoch, i, model.get_current_losses()))
                logger.info('Epoch [{}/{}] iter {}: train HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    e, opt.epoch, i, pad_meter=pad_meter_train))

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
                plt.savefig("%s/epoch_%d.png" % (img_save_dir, e))

                plt.close()
                logger.trace("Epoch [{}] iter {}: save figs ok".format(e, i))

                # vutils.save_image(ret['fake_B'], "%s/epoch_%d_fake.png" % (img_save_dir, e), normalize=True)
                # vutils.save_image(ret['real_B'], "%s/epoch_%d_real.png" % (img_save_dir, e), normalize=True)


        if e%1==0:
            model.eval()
            pad_dev_mater = eval_model(dev_data_loader,model)
            logger.info("Epoch [{}/{}] val_loss: {}".format(e, opt.epoch, model.get_current_losses()))
            writer.add_scalars('val_loss/C', {'C': model.get_current_losses()['C']}, e)
            writer.add_scalars('val_loss/G', {'G_GAN_loss': model.get_current_losses()['G_GAN']}, e)
            writer.add_scalars('val_loss/G', {'G_NP_loss': model.get_current_losses()['G_NP']}, e)
            writer.add_scalars('val_loss/D', {'D_real_loss': model.get_current_losses()['D_real']}, e)
            writer.add_scalars('val_loss/D', {'D_fake_loss': model.get_current_losses()['D_fake']}, e)
            writer.add_scalars('val_loss/D', {'D': model.get_current_losses()['D']}, e)
            writer.add_scalars('val_loss/G', {'G': model.get_current_losses()['G']}, e)

            pad_meter = eval_model(test_data_loader,model)
            logger.info("Epoch [{}/{}] test_loss: {}".format(e, opt.epoch, model.get_current_losses()))
            writer.add_scalars('test_loss/C', {'C': model.get_current_losses()['C']}, e)
            writer.add_scalars('test_loss/G', {'G_GAN_loss': model.get_current_losses()['G_GAN']}, e)
            writer.add_scalars('test_loss/G', {'G_NP_loss': model.get_current_losses()['G_NP']}, e)
            writer.add_scalars('test_loss/D', {'D_real_loss': model.get_current_losses()['D_real']}, e)
            writer.add_scalars('test_loss/D', {'D_fake_loss': model.get_current_losses()['D_fake']}, e)
            writer.add_scalars('test_loss/D', {'D': model.get_current_losses()['D']}, e)
            writer.add_scalars('test_loss/G', {'G': model.get_current_losses()['G']}, e)

            pad_meter.get_eer_and_thr()
            pad_dev_mater.get_eer_and_thr()

            pad_meter.get_hter_apcer_etal_at_thr(pad_dev_mater.threshold)
            pad_meter.get_accuracy(pad_dev_mater.threshold)
            logging.info("epoch %d test"%e)
            logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                pad_meter=pad_meter))
            logger.info('Epoch [{}/{}]: test HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            is_best = pad_meter.hter <= best_res
            best_res = min(pad_meter.hter, best_res)
            if is_best:
                best_name = "best"
                model.save_networks(best_name)

        filename = "lastest"
        model.save_networks(filename)