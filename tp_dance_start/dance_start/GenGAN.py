
import numpy as np
import cv2
import os
import pickle
import sys
import math

import matplotlib.pyplot as plt

from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 



class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        #pass
        return self.model(input)
    



class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False):
        self.netG = GenNNSkeToImage()
        self.netD = Discriminator()
        self.real_label = 1.
        self.fake_label = 0.
        self.filename = 'data/Dance/DanceGenGAN.pth'
        self.filenameOldGen = 'data/Dance/DanceGenVanillaFromSke.pth' # to load the old generator train in GenVanillaNN
        tgt_transform = transforms.Compose(
                            [transforms.Resize((64, 64)),
                            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform)
        print(f"dataset, type(dataset) = {self.dataset},{type(self.dataset)}")
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        # Decide which device we want to run on
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.ngpu > 0) else "cpu")

        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename)
        else:
            self.netG = torch.load(self.filenameOldGen) # load the old generator train in GenVanillaNN for training the new generator in GenGAN


    def train(self, n_epochs=20):
        #pass
        criterion = nn.BCELoss() # Binary Cross Entropy Loss
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.001, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.001, betas=(0.5, 0.999))
        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []

        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(n_epochs):
            print(f"epoch n°={epoch}/{n_epochs}")
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0): # i,(ske,img)
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                ## Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[1].to(self.device)# data[0] is for skeleton, here 1 is for image linked to the skeleton
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through Discriminator
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = criterion(output, label)
                # Calculate gradients for Discriminator in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                ## Train with all-fake batch
                #HERE it's the difference: we don't generate fake image with noise, but from a skeleton
                ske_input = data[0].to(self.device)
                # Generate fake image batch with Generator taking a skeleton as Generator input
                fake = self.netG(ske_input)  # replace the previous instruction self.netG(noise) 
                label.fill_(self.fake_label)
                # Classify all fake batch with Discriminator
                output = self.netD(fake.detach()).view(-1)
                # Calculate Discriminator's loss on the all-fake batch
                errD_fake = criterion(output, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Compute error of D as sum over the fake and the real batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated Discriminator, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                
                # Calculate G's loss based on this output
                errG = criterion(output, label) #-> ensure that G is able to fool the D by generating realistic images
                l1_loss = F.l1_loss(fake, real_cpu) #-> ensure that G generates images (give here with fake) that are close to the real images (give here with real_cpu)
                                                    # measure the pixel-wise difference between the generated image and the real image
                # Total Generator loss
                errG = errG + 10*l1_loss #-> combining losses ensure that G generates images that are realistic and similar to the real images
                                        #-> the 10 is a hyperparameter that can be tuned ; allows to balance the importance of l1_loss
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update Generator
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                        % (epoch, n_epochs, i, len(self.dataloader),
                            errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                
        # Save the new model fo Generator
        torch.save(self.netG, self.filename)
        print("Model netG saved ")
        print("Training is done")
        
        #Plot the training losses as information
        # plt.figure(figsize=(10,5))
        # plt.title("Generator and Discriminator Loss During Training")
        # plt.plot(G_losses,label="G")
        # plt.plot(D_losses,label="D")
        # plt.xlabel("iterations")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.show()
        

    
    def generate(self, ske):           
        """ generator of image from skeleton """
        #pass
        ske_t = torch.from_numpy(ske.reduce().flatten())#ske.__array__(reduced=True).flatten()
        ske_t = ske_t.to(torch.float32)
        ske_t = ske_t.reshape(1,Skeleton.reduced_dim,1,1) # ske.reshape(1,Skeleton.full_dim,1,1)
        normalized_output = self.netG(ske_t)
        res = self.dataset.tensor2image(normalized_output[0])
        return res


if __name__ == '__main__':
    force = False
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    #if False:
    if True:    # train or load
        # Train
        gen = GenGAN(targetVideoSke, False)
        gen.train(20) #20) 5) #200)
    else:
        gen = GenGAN(targetVideoSke, loadFromFile=True)    # load from file        


    for i in range(targetVideoSke.skeCount()):
        image = gen.generate(targetVideoSke.ske[i])
        #image = image*255
        nouvelle_taille = (256, 256) 
        image = cv2.resize(image, nouvelle_taille)
        cv2.imshow('Image', image)
        if cv2.waitKey(-1) == ord('q'):
            break
    cv2.destroyAllWindows()

