import networks
import losses
import torch
from torch import nn
import os
import itertools
from collections import OrderedDict

class FaceModel(nn.Module):
    def __init__(self,opt,isTrain = True,input_nc = 3):
        super(FaceModel,self).__init__()
        self.opt = opt
        self.model = opt.model
        self.w_cls = opt.w_cls 
        self.w_NP = opt.w_NP
        self.w_gan = opt.w_gan
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device(
            'cpu')  # get device name: CPU or GPU
        torch.backends.cudnn.benchmark = True
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.isTrain = isTrain
        self.netEncoder = networks.init_net(networks.Encoder32(),gpu_ids=self.gpu_ids)
        self.netClassifier = networks.init_net(networks.Classifier(), gpu_ids=self.gpu_ids)
        self.netSigDecoder = networks.init_net(networks.Decoder(),gpu_ids=self.gpu_ids)
        self.netSigDiscriminator = networks.init_net(networks.Discriminator(),gpu_ids=self.gpu_ids)

        self.model_names = ["Encoder","SigDecoder","SigDiscriminator","Classifier"]
        self.visual_names = ["real_A","real_B","fake_B"]
        self.loss_names = ['G_GAN', 'G_NP', 'D_real', 'D_fake','C']

        self.channels = 3
        self.frames = 64
        self.width = 128
        self.height = 128

        if self.isTrain:


            # Discriminator loss
            self.criterionGan = losses.GANLoss().to(self.device)
            # Decoder loss
            self.criterionNP = losses.Neg_Pearson().to(self.device)
            # self.criterionL1 = torch.nn.L1Loss()
            # cls loss
            self.criterionCls = [torch.nn.CrossEntropyLoss(),losses.FocalLoss()]
            # net G/
            self.optimizer_sig = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(),
                                                                    self.netSigDecoder.parameters()), lr=opt.lr,betas=(opt.beta1, 0.999))

            # net D/
            self.optimizer_discriminate = torch.optim.Adam(self.netSigDiscriminator.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

            # net cls 
            self.optimizer_cls = torch.optim.Adam(itertools.chain(self.netEncoder.parameters(),
                                                                    self.netClassifier.parameters()),lr=opt.lr, betas=(opt.beta1, 0.999),weight_decay=0.01)

    def set_input(self,input):
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.label = torch.tensor(input['label']).to(self.device)
        self.image_path = input['A_paths']
        
        bs, self.channels, self.frames, self.height, self.width = input['A'].shape

        print("set_input batch: {}".format(bs))

    def forward(self):
        print("-------- FORWARD -----------")
        # torch.autograd.set_detect_anomaly(True)
        # for param_tensor in self.netEncoder.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        #     print(param_tensor,'\t',self.netEncoder.state_dict()[param_tensor])
        
        # print("In forward(), real_A: {}".format(self.real_A.shape))
        # print("real_A: {}".format(self.real_A))
        self.lantent = self.netEncoder(self.real_A)
        # print("In forward(), lantent: {}".format(self.lantent.shape))
        # print("lantent: {}".format(self.lantent))
        self.fake_B = self.netSigDecoder(self.lantent)
        # print("fake_B: {}".format(self.fake_B))
        # print("In forward(), fake_B: {}".format(self.fake_B.shape))
        self.output = self.netClassifier(self.lantent)


    def backward_D(self):
        fake_B_repeated = torch.repeat_interleave(self.fake_B, self.width*self.height, dim=1).view(-1, 1, self.frames, self.height, self.width)        # [B, frames] -> [B, 1, frames, 128, 128], repeat to cat
        fake_AB = torch.cat((self.real_A, fake_B_repeated), 1)       # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]
        pred_fake = self.netSigDiscriminator(fake_AB.detach())
        self.loss_D_fake = self.criterionGan(pred_fake,False)

        real_B_repeated = torch.repeat_interleave(self.real_B, self.width*self.height, dim=1).view(-1, 1, self.frames, self.height, self.width)        # [B, frames] -> [B, 1, frames, 128, 128], repeat to cat
        real_AB = torch.cat((self.real_A, real_B_repeated), 1)       # [B, 3, frames, 128, 128] + [B, 1, frames, 128, 128] -> [B, 4, frames, 128, 128]
        # print("!!!!!!!!!!!!!!!!!!!!!!! real_AB dtype: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(real_AB.dtype))
        pred_real = self.netSigDiscriminator(real_AB.detach())
        self.loss_D_real = self.criterionGan(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 *self.w_gan
        print("loss_D_fake: {}, loss_D_real: {}, loss_D: {}".format(self.loss_D_fake, self.loss_D_real, self.loss_D))
        self.loss_D.backward()

    def backward_G(self):
        fake_B_repeated = torch.repeat_interleave(self.fake_B, self.width*self.height, dim=1).view(-1, 1, self.frames, self.height, self.width)        # [B, 1, frames] -> [B, 1, frames, 128, 128], repeat to cat
        fake_AB = torch.cat((self.real_A, fake_B_repeated), 1)
        pred_fake = self.netSigDiscriminator(fake_AB.detach())
        self.loss_G_GAN = self.criterionGan(pred_fake, True)
        self.loss_G_NP = self.criterionNP(self.fake_B, self.real_B)
        self.loss_G = self.loss_G_NP*self.w_NP + self.loss_G_GAN *self.w_gan
        print("loss_G_GAN: {}, loss_G_NP: {}, loss_G: {}".format(self.loss_G_GAN, self.loss_G_NP, self.loss_G))
        self.loss_G.backward()


    def backward_C(self):
        output = self.output
        self.loss_C = (2* self.criterionCls[0](output,self.label)+  self.criterionCls[1](output,self.label))*self.w_cls #self.criterionCls[0](output,self.label)+  self.criterionCls[1](cls_feat,self.label)
        print("loss_C: {}".format(self.loss_C))
        self.loss_C.backward()

    def optimize_parameters(self):
        self.forward()
        
        if self.model =="model3":
            # update D
            print("-------- UPDATE D -------------")
            self.set_requires_grad(self.netSigDiscriminator, True) 
            self.optimizer_discriminate.zero_grad()
            # with torch.autograd.detect_anomaly():
            #     self.backward_D()
            self.backward_D()
            for name, param in self.netSigDiscriminator.named_parameters():
                print(name, torch.isnan(param.grad).all())
            self.optimizer_discriminate.step()
            self.optimizer_discriminate.zero_grad()
        if self.model =="model3" or self.model =="model2":
            # update G_depth
            print("-------- UPDATE G -------------")
            self.set_requires_grad(self.netSigDiscriminator, False) 
            self.optimizer_sig.zero_grad()
            # with torch.autograd.detect_anomaly():
            #     self.backward_G()
            self.backward_G() 
            for name, param in self.netEncoder.named_parameters():
                print(name, torch.isnan(param.grad).all())
            for name, param in self.netSigDecoder.named_parameters():
                print(name, torch.isnan(param.grad).all())
            self.optimizer_sig.step()
            self.optimizer_sig.zero_grad()
        if self.model =="model3" or self.model =="model2" or self.model =="model1":
            print("-------- UPDATE C -------------")
            self.forward()
            # update C
            self.optimizer_cls.zero_grad()
            self.backward_C()
            for name, param in self.netClassifier.named_parameters():
                print(name, torch.isnan(param.grad).all())
            self.optimizer_cls.step()
            self.optimizer_cls.zero_grad()

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
                net.train()

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
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
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