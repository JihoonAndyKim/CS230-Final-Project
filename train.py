import os
import time
import copy

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as ds
import torchvision.transforms as transforms

import models
import iotools

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Hyperparameters
epochs         = 250  # Epochs
learning_rate  = 0.002 # Learning rate
beta1          = 0.5   # Exp. weight decay
beta2          = 0.999 # Exp. weight decay
cuda_flag      = False  # Set this to true to train with GPUs

# Main CycleGAN class
class CycleGAN():

    def load_data(self, data_A, data_B, test_A, test_B):

		# Iterators for data files
        self.data_A = iter(data_A)
        self.data_B = iter(data_B)
        self.test_A = iter(test_A)
        self.test_B = iter(test_B)

		# Retrieve number of image files
        self.data_A_size = len(self.data_A)
        self.data_B_size = len(self.data_B)
        self.test_A_size = len(self.test_A)
        self.test_B_size = len(self.test_B)

		# Set number of images
        self.num_images = min(self.data_A_size, self.data_B_size)

    def init_model(self):

		# Construct Generator / Discriminator NN
        self.DiscA = models.Discriminator_5()
        self.DiscB = models.Discriminator_5()
        self.GenA  = models.Generator(9)      # Generator with 9 ResNets
        self.GenB  = models.Generator(9)      # Generator with 9 ResNets

        # Make CUDA tensors
        if cuda_flag:
            device = torch.device("cuda:0")
            num_device = torch.cuda.device_count()
            print("Number of devices: ", num_device)
            self.DiscA = nn.DataParallel(self.DiscA, device_ids = [i for i in range(num_device)])
            self.DiscB = nn.DataParallel(self.DiscB, device_ids = [i for i in range(num_device)])
            self.GenA = nn.DataParallel(self.GenA, device_ids = [i for i in range(num_device)])
            self.GenB = nn.DataParallel(self.GenB, device_ids = [i for i in range(num_device)])

            print("Sending models to device")
            self.DiscA = self.DiscA.to(device)
            self.DiscB = self.DiscB.to(device)
            self.GenA  = self.GenA.to(device)
            self.GenB  = self.GenB.to(device)

        # Optimizer
        print("Optimizer init")
        self.DiscA_opt = torch.optim.Adam(self.DiscA.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.DiscB_opt = torch.optim.Adam(self.DiscB.parameters(), lr=learning_rate, betas=(beta1, beta2))
        self.GenA_opt  = torch.optim.Adam(self.GenA.parameters(),  lr=learning_rate, betas=(beta1, beta2))
        self.GenB_opt  = torch.optim.Adam(self.GenB.parameters(),  lr=learning_rate, betas=(beta1, beta2))

    def checkpoint_load(self):
        print("Loading checkpoint")
		# Create checkpoint directory if not present
        self.checkpoint_dir = './checkpoint/leather2denim'
        iotools.create_dirs(self.checkpoint_dir)
        try:
	    	# Retrieve checkpointing if present
            ckpt = iotools.checkpoint_load(self.checkpoint_dir)
	    	# Generator
            self.GenA.load_state_dict(ckpt['GenA'])
            self.GenB.load_state_dict(ckpt['GenB'])
            self.GenA_opt.load_state_dict(ckpt['GenA_opt'])
            self.GenB_opt.load_state_dict(ckpt['GenB_opt'])
            # Discriminator
            self.DiscA.load_state_dict(ckpt['DiscA'])
            self.DiscB.load_state_dict(ckpt['DiscB'])
            self.DiscA_opt.load_state_dict(ckpt['DiscA_opt'])
            self.DiscB_opt.load_state_dict(ckpt['DiscB_opt'])
	    	# Epoch
            self.curr_epoch = ckpt['epoch']
        except:
            self.curr_epoch = 0
            print('No checkpoint available')


    def train(self):
		# Create pool of images
        fake_pool_A = ImagePool()
        fake_pool_B = ImagePool()
        print("Training started")
		# Training loop
        for epoch in range(self.curr_epoch, epochs):
            for i, (input_A, input_B) in enumerate(zip(self.data_A, self.data_B)):

                step = epoch * self.num_images + (i + 1)
                print("Epoch: " + str(epoch) + ", Step: " + str(i + 1) + " out of " + str(self.num_images))

				# Timing
                start = time.time()

                # Train Generators (GenA + GenB) -------------------------------
                self.GenA.train()
                self.GenB.train()

                # Put image in variable
                input_A = input_A[0]
                input_B = input_B[0]
                input_A = Variable(input_A)
                input_B = Variable(input_B)

                if cuda_flag:
                    input_A = input_A.cuda()
                    input_B = input_B.cuda()

                # Generate images
                fake_A = self.GenA(input_A)
                fake_B = self.GenB(input_B)
                cyc_A  = self.GenA(fake_B)
                cyc_B  = self.GenB(fake_A)

                # Discriminate fake images
                fake_rec_A = self.DiscA(fake_A)
                fake_rec_B = self.DiscB(fake_B)

                # Compute losses
                MSELoss = nn.MSELoss()

				# Create ones tensor for generator MSELoss
                if cuda_flag:
                    onesA   = Variable(torch.ones(fake_rec_A.size())).cuda()
                    onesB   = Variable(torch.ones(fake_rec_B.size())).cuda()
                else:
                    onesA    = Variable(torch.ones(fake_rec_A.size()))
                    onesB    = Variable(torch.ones(fake_rec_B.size()))

                disc_A_loss = MSELoss(fake_rec_A, onesA)
                disc_B_loss = MSELoss(fake_rec_B, onesB)

                # Cyclic loss
                L1Loss = nn.L1Loss()
                cyc_A_loss = L1Loss(cyc_A, input_A)
                cyc_B_loss = L1Loss(cyc_B, input_B)
                cyc_loss   = cyc_A_loss + cyc_B_loss

                # Computing generator loss
                gen_A_loss = cyc_loss*10 + disc_B_loss
                gen_B_loss = cyc_loss*10 + disc_A_loss
                gen_loss = gen_A_loss + gen_B_loss


                # Train Discriminator (DiscA + DiscB) --------------------------
                fake_A = Variable(torch.Tensor(fake_pool_A.getImage(fake_A.cpu().data.numpy())))
                fake_B = Variable(torch.Tensor(fake_pool_B.getImage(fake_B.cpu().data.numpy())))

                if cuda_flag:
                    fake_A.cuda()
                    fake_B.cuda()

                # Training the discriminator
                disc_true_A = self.DiscA(input_A)
                disc_fake_A = self.DiscA(fake_A)
                disc_true_B = self.DiscB(input_B)
                disc_fake_B = self.DiscB(fake_B)

				# Create ones/zeros tensor for discriminator MSELoss
                if cuda_flag:
                    onesA    = Variable(torch.ones(disc_fake_A.size())).cuda()
                    onesB    = Variable(torch.ones(disc_fake_B.size())).cuda()
                    zerosA   = Variable(torch.ones(disc_fake_A.size())).cuda()
                    zerosB   = Variable(torch.ones(disc_fake_B.size())).cuda()
                else:
                    onesA    = Variable(torch.ones(disc_fake_A.size()))
                    onesB    = Variable(torch.ones(disc_fake_B.size()))
                    zerosA   = Variable(torch.ones(disc_fake_A.size()))
                    zerosB   = Variable(torch.ones(disc_fake_B.size()))

                # Computing discriminator loss
                disc_true_A_loss = MSELoss(disc_true_A, onesA)
                disc_fake_A_loss = MSELoss(disc_fake_A, zerosA)
                disc_true_B_loss = MSELoss(disc_true_B, onesB)
                disc_fake_B_loss = MSELoss(disc_fake_B, zerosB)

                disc_A_loss = disc_true_A_loss + disc_fake_A_loss
                disc_B_loss = disc_true_B_loss + disc_fake_B_loss


                # Backward propagation Discriminator (DiscA + DiscB) -----------

                # Generator backwarddisc_A_loss
                self.GenA.zero_grad()
                self.GenB.zero_grad()
                gen_loss.backward()
                self.GenA_opt.step()
                self.GenB_opt.step()

                # Discriminator backward
                self.DiscA.zero_grad()
                self.DiscB.zero_grad()
                disc_A_loss.backward()
                disc_B_loss.backward()
                self.DiscA_opt.step()
                self.DiscB_opt.step()

                #Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(self.DiscA.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.DiscB.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.GenA.parameters(), 5)
                torch.nn.utils.clip_grad_norm_(self.GenB.parameters(), 5)

                # Report results -----------------------------------------------

				# Timing
                end = time.time()
                print("Time for step ",i,": ", end - start)

                if (i + 1) % 1 == 0:
                    message = "(epoch: " + str(epoch) + ", iters: " + str(i) + ", DiscA_loss: " + str(disc_A_loss.item()) + ", DiscB_loss: " + str(disc_B_loss.item()) \
                                                    + ", GenA_loss: " + str(gen_A_loss.item()) + ", GenB_loss: " + str(gen_B_loss.item()) + ", cycA_loss: " + str(cyc_A_loss.item()) \
                                                    + ", cycB_loss: " + str(cyc_B_loss.item())
                    print(message)
                    self.GenA.eval()
                    self.GenB.eval()

                    test_A_output = Variable(self.test_A.next()[0])
                    test_B_output = Variable(self.test_B.next()[0])

                    if cuda_flag:
                        test_A_output = test_A_output.cuda()
                        test_B_output = test_B_output.cuda()

                    test_A_fake = self.GenA(test_B_output)
                    test_B_fake = self.GenB(test_A_output)

                    A_rec_test = self.GenA(test_B_fake)
                    B_rec_test = self.GenB(test_A_fake)

                    test_A_output   = test_A_output.cpu()
                    test_B_fake     = test_B_fake.cpu()
                    A_rec_test      = A_rec_test.cpu()
                    test_B_output   = test_B_output.cpu()
                    test_A_fake     = test_A_fake.cpu()
                    B_rec_test      = B_rec_test.cpu()

                    save_dir = './images'
                    iotools.create_dirs(save_dir)
                    torchvision.utils.save_image((test_A_output + 1) / 2.0, '%s/Epoch%d_image%d_test_A.jpg' % (save_dir, epoch, i + 1), nrow=3)
                    torchvision.utils.save_image((test_B_fake + 1) / 2.0, '%s/Epoch%d_image%d_test_B_fake.jpg' % (save_dir, epoch, i + 1), nrow=3)
                    torchvision.utils.save_image((A_rec_test + 1) / 2.0, '%s/Epoch%d_image%d_a_recovered.jpg' % (save_dir, epoch, i + 1), nrow=3)
                    torchvision.utils.save_image((test_B_output + 1) / 2.0, '%s/Epoch%d_image%d_test_B.jpg' % (save_dir, epoch, i + 1), nrow=3)
                    torchvision.utils.save_image((test_A_fake + 1) / 2.0, '%s/Epoch%d_image%d_test_A_fake.jpg' % (save_dir, epoch, i + 1), nrow=3)
                    torchvision.utils.save_image((B_rec_test + 1) / 2.0, '%s/Epoch%d_image%d_b_recovered.jpg' % (save_dir, epoch, i + 1), nrow=3)
                    print("Saved images")
                    
                    f = open("Losses.txt",'w')
                    f.write(message)
                    f.close()

                    iotools.checkpoint_save({'epoch': epoch + 1,
                 	                         'DiscA': self.DiscA.state_dict(),
                                             'DiscB': self.DiscB.state_dict(),
                                             'GenA': self.GenA.state_dict(),
                                             'GenB': self.GenB.state_dict(),
                                             'DiscA_opt': self.DiscA_opt.state_dict(),
                                             'DiscB_opt': self.DiscB_opt.state_dict(),
                                             'GenA_opt': self.GenA_opt.state_dict(),
                                             'GenB_opt': self.GenB_opt.state_dict()},
                                             '%s/Epoch_(%d).ckpt' % (self.checkpoint_dir, epoch + 1))

# Image pool to randomize Discriminator input
class ImagePool():
    def __init__(self, max_images = 50):
        self.max_images = max_images
        self.curr_num = 0
        self.images = []

    def getImage(self, image):
		# If pool not full, return input image
        if self.curr_num < self.max_images:
            self.images.append(image)
            self.curr_num += 1
            return image
        else:
            if np.random.uniform() > 0.5:
                index = np.random.randint(0, self.max_images)
                temp = copy.copy(self.images[index])
                self.images[index] = image
                return temp
            else:
                return image

######################################################################################
# MAIN

# Retrieve GPU environment
if cuda_flag:
    num_device = torch.cuda.device_count()
    gpu_ids = [str(i) for i in range(num_device)]
else:
    gpu_ids = [str(i) for i in range(1)]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

#Data augmentation
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.Resize(286),
     transforms.RandomCrop(256),
     transforms.RandomAffine(15),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])

data_dir = './Data/'
data_dir = iotools.create_fs_structure(data_dir)

# Apply data augmentation to datasets
data_A = ds.ImageFolder(data_dir['trainA'], transform=transform)
data_B = ds.ImageFolder(data_dir['trainB'], transform=transform)
data_A_test = ds.ImageFolder(data_dir['testA'], transform=transform)
data_B_test = ds.ImageFolder(data_dir['testB'], transform=transform)

# Load images (one at a time per GPU) with shuffling
if cuda_flag:
    A_loader = torch.utils.data.DataLoader(data_A, batch_size=1, shuffle=True, num_workers=num_device)
    B_loader = torch.utils.data.DataLoader(data_B, batch_size=1, shuffle=True, num_workers=num_device)
    A_test_loader = torch.utils.data.DataLoader(data_A_test, batch_size=3, shuffle=True, num_workers=num_device)
    B_test_loader = torch.utils.data.DataLoader(data_B_test, batch_size=3, shuffle=True, num_workers=num_device)
else:
    A_loader = torch.utils.data.DataLoader(data_A, batch_size=1, shuffle=True, num_workers=1)
    B_loader = torch.utils.data.DataLoader(data_B, batch_size=1, shuffle=True, num_workers=1)
    A_test_loader = torch.utils.data.DataLoader(data_A_test, batch_size=3, shuffle=True, num_workers=1)
    B_test_loader = torch.utils.data.DataLoader(data_B_test, batch_size=3, shuffle=True, num_workers=1)

# Create model
gan = CycleGAN()

# Load data
gan.load_data(A_loader, B_loader, A_test_loader, B_test_loader)

# Init model
gan.init_model()

# Check whether a checkpoint exists and load it
gan.checkpoint_load()

# Train the network
gan.train()
#################################################################################
