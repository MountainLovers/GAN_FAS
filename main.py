from dataset import AlignedDataset
from torch.utils.data import DataLoader
from torch import nn
from model import FaceModel
from options import opt
import torchvision.utils as vutils
import os
import torch
from statistics import PADMeter
import logging
from tensorboardX import SummaryWriter
from torch.utils.data.sampler import  WeightedRandomSampler
from test import eval_model
import sys
from loguru import logger

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

if __name__ == '__main__':
    best_res = 101
    train_batch_size = opt.batch_size
    test_batch_size = opt.batch_size
    
    train_file_list = opt.train_file_list
    dev_file_list = opt.dev_file_list
    test_file_list = opt.test_file_list
    model = FaceModel(opt,isTrain = True,input_nc = 3)
    test_data_loader = DataLoader(AlignedDataset(test_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=8)
    dev_data_loader = DataLoader(AlignedDataset(dev_file_list,isTrain = False), batch_size=test_batch_size,
                                   shuffle=True, num_workers=8)

    train_dataset = AlignedDataset(train_file_list) 
    train_data_loader = DataLoader(train_dataset, batch_size=train_batch_size,
                                    shuffle = True,num_workers=8, drop_last=True)

    loss_names = model.loss_names

    writer.iter = 0
    for e in range(opt.epoch):
        logger.info("EPOCH [{}/{}] TRAIN".format(e, opt.epoch))
        model.train()
        pad_meter_train = PADMeter()
        itern = len(train_data_loader)
        for i, data in enumerate(train_data_loader):
            model.set_input(data)
            model.optimize_parameters()
            class_output = nn.functional.softmax(model.output, dim=1)
            pad_meter_train.update(model.label.cpu().data.numpy(),
                             class_output.cpu().data.numpy())
            for name in loss_names:
                writer.add_scalars("train_loss/"+name[0], {name: model.get_current_losses()[name]}, i+ len(train_data_loader) *e)

            if i %100 ==0:
                pad_meter_train.get_eer_and_thr()
                pad_meter_train.get_hter_apcer_etal_at_thr(pad_meter_train.threshold)
                pad_meter_train.get_accuracy(pad_meter_train.threshold)
                ret = model.get_current_visuals()
                img_save_dir = os.path.join(opt.checkpoints_dir, opt.name, "res")
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                logger.info("Epoch [{}/{}] - TRAIN iter [{}/{}]: loss: {}".format(e, opt.epoch, i, itern, model.get_current_losses()))
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    e, opt.epoch, i, itern, pad_meter=pad_meter_train))
                logger.info('Epoch [{}/{}] - TRAIN iter [{}/{}]: APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
                    e, opt.epoch, i, itern, pad_meter=pad_meter_train))

                logging.info(model.get_current_losses())
                logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                    pad_meter=pad_meter_train))
                vutils.save_image(ret['fake_B'], "%s/epoch_%d_fake.png" % (img_save_dir, e), normalize=True)
                vutils.save_image(ret['real_B'], "%s/epoch_%d_real.png" % (img_save_dir, e), normalize=True)


        if e%1==0:
            logger.info("EPOCH [{}/{}] EVAL".format(e, opt.epoch))
            model.eval()
            pad_dev_mater = eval_model(dev_data_loader,model)
            pad_meter = eval_model(test_data_loader,model)

            pad_meter.get_eer_and_thr()
            pad_dev_mater.get_eer_and_thr()

            pad_meter.get_hter_apcer_etal_at_thr(pad_dev_mater.threshold)
            pad_meter.get_accuracy(pad_dev_mater.threshold)
            logger.info('Epoch [{}/{}] - TEST: HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            logger.info('Epoch [{}/{}] - TEST: APCER {pad_meter.apcer:.4f} BPCER {pad_meter.bpcer:.4f} ACER {pad_meter.acer:.4f} AUC {pad_meter.auc:.4f}'.format(
                e, opt.epoch, pad_meter=pad_meter))
            logging.info("epoch %d"%e)
            logging.info('HTER {pad_meter.hter:.4f} EER {pad_meter.eer:.4f} ACC {pad_meter.accuracy:.4f}'.format(
                pad_meter=pad_meter))

            ret = model.get_current_visuals()
            vutils.save_image(ret['fake_B'], "%s/epoch_%d_fake_eval.png" % (img_save_dir, e), normalize=True)
            vutils.save_image(ret['real_B'], "%s/epoch_%d_real_eval.png" % (img_save_dir, e), normalize=True)

            is_best = pad_meter.hter <= best_res
            best_res = min(pad_meter.hter, best_res)
            if is_best:
                best_name = "best"
                model.save_networks(best_name)

        filename = "lastest"
        model.save_networks(filename)