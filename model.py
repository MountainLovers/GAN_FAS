import networks
import losses
import torch
from torch import nn
import torch.autograd as autograd
import os
import itertools
from collections import OrderedDict
from loguru import logger
import sys

LAMBDA = 10 # Gradient penalty lambda hyperparameter

class FaceModel(nn.Module):
    def __init__(self,opt,isTrain = True,input_nc = 3):
        super(FaceModel,self).__init__()
        self.opt = opt

        self.model = opt.model
        self.w_cls = opt.w_cls 
        self.w_NP = opt.w_NP
        self.w_L1 = opt.w_L1
        self.w_gan = opt.w_gan

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        # torch.backends.cudnn.benchmark = True
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.isTrain = isTrain
        self.netEncoder = networks.init_net(networks.Encoder32(),gpu_ids=self.gpu_ids)
        self.netClassifier = networks.init_net(networks.Classifier(), gpu_ids=self.gpu_ids)
        self.netSigDecoder = networks.init_net(networks.Decoder(),gpu_ids=self.gpu_ids)
        self.netSigDiscriminator = networks.init_net(networks.Discriminator(),gpu_ids=self.gpu_ids)

        self.model_names = ["Encoder","SigDecoder","SigDiscriminator","Classifier"]
        self.sig_names = ["real_B","fake_B"]
        self.loss_names = ['G_GAN', 'G_NP', 'G_L1', 'D_real', 'D_fake','C', 'D', 'G']
        self.val_loss_names = ['G_GAN', 'G_NP', 'G_L1', 'D_real', 'D_fake','C', 'D', 'G']

        self.batch_size = opt.batch_size
        self.channels = 3
        self.frames = 64
        self.width = 128
        self.height = 128

        # Discriminator loss
        self.criterionGan = losses.GANLoss()
        # Decoder loss
        self.criterionNP = losses.Neg_Pearson()
        self.criterionL1 = torch.nn.L1Loss()
        # cls loss
        self.criterionCls = [torch.nn.CrossEntropyLoss(),losses.FocalLoss()]
        
        if self.isTrain:
            # net G/
            self.optimizer_sig = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(),
                                                                    self.netSigDecoder.parameters()), lr=opt.lr_G,betas=(opt.beta1, 0.999))

            # net D/
            self.optimizer_discriminate = torch.optim.Adam(self.netSigDiscriminator.parameters(),lr=opt.lr_D, betas=(opt.beta1, 0.999))

            # net cls 
            self.optimizer_cls = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(),
                                                                    self.netClassifier.parameters()),lr=opt.lr_C, betas=(opt.beta1, 0.999),weight_decay=0.01)

    def set_input(self,input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_path = input['A_paths']
        
        self.bs, self.channels, self.frames, self.height, self.width = input['A'].shape

        # logger.debug("set_input batch: {}".format(self.bs))

    def forward(self):
        logger.trace("FORWARD")
        # torch.autograd.set_detect_anomaly(True)
        # for param_tensor in self.netEncoder.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor,'\t',self.netEncoder.state_dict()[param_tensor])
        
        # print("In forward(), real_A: {}".format(self.real_A.shape))
        # print("real_A: {}".format(self.real_A))
        self.lantent = self.netEncoder(self.real_A)
        # logger.info("In forward(), lantent: {}".format(self.lantent.shape))
        # print("lantent: {}".format(self.lantent))
        self.fake_B = self.netSigDecoder(self.lantent)
        # print("fake_B: {}".format(self.fake_B))
        # logger.info("In forward(), fake_B: {}".format(self.fake_B.shape))
        self.output = self.netClassifier(self.lantent)

        fake_B_repeated = self.fake_B.reshape(self.bs, 1, self.frames, 1, 1).expand(self.bs, 1, self.frames, self.height, self.width)
        self.fake_AB = torch.cat((self.real_A, fake_B_repeated), 1)       # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]
        real_B_repeated = self.real_B.reshape(self.bs, 1, self.frames, 1, 1).expand(self.bs, 1, self.frames, self.height, self.width)
        self.real_AB = torch.cat((self.real_A, real_B_repeated), 1)       # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]

    def backward_D(self):
        # fake_B_repeated = self.fake_B.reshape(self.bs, 1, self.frames, 1, 1).expand(self.bs, 1, self.frames, self.height, self.width)
        # fake_AB = torch.cat((self.real_A, fake_B_repeated), 1)       # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]
        pred_fake = self.netSigDiscriminator(self.fake_AB.detach())
        self.loss_D_fake = self.criterionGan(pred_fake, False)

        # real_B_repeated = self.real_B.reshape(self.bs, 1, self.frames, 1, 1).expand(self.bs, 1, self.frames, self.height, self.width)
        # real_AB = torch.cat((self.real_A, real_B_repeated), 1)       # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]
        pred_real = self.netSigDiscriminator(self.real_AB.detach())
        self.loss_D_real = self.criterionGan(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.w_gan
        logger.debug("loss_D_fake: {}, loss_D_real: {}, loss_D: {}".format(self.loss_D_fake.item(), self.loss_D_real.item(), self.loss_D.item()))
        self.loss_D.backward()

        # train with gradient penalty
        gradient_penalty = self.calc_gradient_penalty(self.real_AB.detach(), self.fake_AB.detach())
        logger.debug("gradient_penalty: {}".format(gradient_penalty.item()))
        gradient_penalty.backward()

    def backward_G(self):
        # fake_B_repeated = self.fake_B.reshape(self.bs, 1, self.frames, 1, 1).expand(self.bs, 1, self.frames, self.height, self.width)
        # fake_AB = torch.cat((self.real_A, fake_B_repeated), 1)
        pred_fake = self.netSigDiscriminator(self.fake_AB.detach())
        self.loss_G_GAN = self.criterionGan(pred_fake, True)
        self.loss_G_NP = self.criterionNP(self.fake_B, self.real_B)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_NP*self.w_NP + self.loss_G_GAN *self.w_gan + self.loss_G_L1*self.w_L1
        logger.debug("loss_G_GAN: {}, loss_G_NP: {}, loss_G_L1: {}, loss_G: {}".format(self.loss_G_GAN.item(), self.loss_G_NP.item(), self.loss_G_L1.item(), self.loss_G.item()))
        # logger.debug("loss_G backward start")
        self.loss_G.backward()


    def backward_C(self):
        output = self.output
        self.loss_C = (2* self.criterionCls[0](output,self.label)+  self.criterionCls[1](output,self.label))*self.w_cls #self.criterionCls[0](output,self.label)+  self.criterionCls[1](cls_feat,self.label)
        logger.debug("loss_C: {}".format(self.loss_C.item()))
        self.loss_C.backward()

    def optimize_parameters(self):
        self.forward()
        
        if self.model =="model3":
            # update D
            logger.trace("UPDATE D")
            self.set_requires_grad(self.netSigDiscriminator, True) 
            self.optimizer_discriminate.zero_grad()
            # with torch.autograd.detect_anomaly():
            #     self.backward_D()
            # logger.trace("UPDATE D backward_D start")
            self.backward_D()
            # logger.trace("UPDATE D backward_D ok")
            # for name, param in self.netSigDiscriminator.named_parameters():
            #     print(name, torch.isnan(param.grad).all())
            self.optimizer_discriminate.step()
            # self.optimizer_discriminate.zero_grad()
        if self.model =="model3" or self.model =="model2":
            # update G_depth
            logger.trace("UPDATE G")
            self.set_requires_grad(self.netSigDiscriminator, False) 
            self.optimizer_sig.zero_grad()
            # with torch.autograd.detect_anomaly():
            #     self.backward_G()
            # logger.trace("UPDATE G backward_G start")
            self.backward_G() 
            # logger.trace("UPDATE G backward_G ok")
            # for name, param in self.netEncoder.named_parameters():
            #     print(name, torch.isnan(param.grad).all())
            # for name, param in self.netSigDecoder.named_parameters():
            #     print(name, torch.isnan(param.grad).all())
            self.optimizer_sig.step()
            # self.optimizer_sig.zero_grad()
        if self.model =="model3" or self.model =="model2" or self.model =="model1":
            logger.trace("UPDATE C")
            self.forward()
            # update C
            self.optimizer_cls.zero_grad()
            self.backward_C()
            # for name, param in self.netClassifier.named_parameters():
            #     print(name, torch.isnan(param.grad).all())
            self.optimizer_cls.step()
            # self.optimizer_cls.zero_grad()
    
    def cal_loss(self):
        ret = {}
        # D    # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]
        pred_fake = self.netSigDiscriminator(self.fake_AB.detach())
        val_loss_D_fake = self.criterionGan(pred_fake, False)
        pred_real = self.netSigDiscriminator(self.real_AB.detach())
        val_loss_D_real = self.criterionGan(pred_real, True)
        val_loss_D = (val_loss_D_fake + val_loss_D_real) * 0.5 *self.w_gan
        # logger.debug("VAL: loss_D_fake: {}, loss_D_real: {}, loss_D: {}".format(val_loss_D_fake.item(), val_loss_D_real.item(), val_loss_D.item(), val_gradient_penalty.item()))
        ret['D_fake'] = val_loss_D_fake.item()
        ret['D_real'] = val_loss_D_real.item()
        ret['D'] = val_loss_D.item()

        # G
        pred_fake = self.netSigDiscriminator(self.fake_AB.detach())
        val_loss_G_GAN = self.criterionGan(pred_fake, True)
        val_loss_G_NP = self.criterionNP(self.fake_B, self.real_B)
        val_loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        val_loss_G = val_loss_G_NP*self.w_NP + val_loss_G_GAN * self.w_gan + val_loss_G_L1*self.w_L1
        # logger.debug("VAL: loss_G_GAN: {}, loss_G_NP: {}, loss_G: {}".format(val_loss_G_GAN.item(), val_loss_G_NP.item(), val_loss_G.item()))
        ret['G_GAN'] = val_loss_G_GAN.item()
        ret['G_NP'] = val_loss_G_NP.item()
        ret['G_L1'] = val_loss_G_L1.item()
        ret['G'] = val_loss_G.item()

        # C
        output = self.output.detach()
        val_loss_C = (2* self.criterionCls[0](output,self.label)+  self.criterionCls[1](output,self.label))*self.w_cls #self.criterionCls[0](output,self.label)+  self.criterionCls[1](cls_feat,self.label)
        # logger.debug("VAL: loss_C: {}".format(val_loss_C.item()))
        ret['C'] = val_loss_C.item()

        return ret

    def calc_gradient_penalty(self, real_data, fake_data):
        #print real_data.size()
        alpha = torch.rand(self.batch_size, 1, 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device) if self.gpu_ids else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if self.gpu_ids:
            interpolates = interpolates.to(self.device)
        # interpolates = autograd.Variable(interpolates, requires_grad=True)
        interpolates.requires_grad_(True)

        disc_interpolates = self.netSigDiscriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device) if self.gpu_ids else torch.ones(
                                    disc_interpolates.size()),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def eval(self):
        self.isTrain = False
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        self.isTrain = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                logger.trace(name + " train start")
                net.train()
                logger.trace(name + " train ok")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                logger.info('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

    def get_current_sigs(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.sig_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
                visual_ret[name] = visual_ret[name].clone().detach().cpu().numpy()
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                try:
                    errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
                except Exception as e:
                    errors_ret[name] = -1
        return errors_ret

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)