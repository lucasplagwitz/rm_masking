import argparse
import logging
import os
import numpy as np
from skimage import color
import nibabel as nib
import scipy.ndimage
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from rm_masking.utils.data_loading import BasicDataset
from rm_masking.unet import UNet


def dice_score(mask, gt):
    return np.sum(mask[gt == 1])*2 / (np.sum(mask) + np.sum(gt))

model_path = "../models/"

mouse_test_img_path = "../../data/png/mouse_test/img/"
mouse_test_mask_path = "../../data/png/mouse_test/mask/"

amount = [str(x) for x in [5, 10, 20, 30, 40, 50]] + ["58", "64*"]
models = [f"mouse_{x}.pth" for x in amount[:-2]] + ["only_mouse.pth", "mouse_rat.pth"]

all_sum_mouse = []
for i, model in enumerate(models):
    #net = UNet(n_channels=1, n_classes=2)

    #if i in [0, 1, 2, 3, 7]:
    net = UNet(n_channels=1, n_classes=2, bilinear=False)

    device = "cuda:0"
    net.to(device=device)
    net.load_state_dict(torch.load(model_path+model, map_location=device))

    dataset_test = BasicDataset(mouse_test_img_path, mouse_test_mask_path, 1)

    sum = 0
    for j in range(len(dataset_test.ids)):
        #a = net(dataset_test[j]["image"].reshape((1, 1, 80, 80)).to(device='cuda')).cpu().detach().numpy()

        m = np.asarray(Image.open(mouse_test_img_path + dataset_test.ids[j] + ".png").convert('L')) / 255
        a=net(torch.from_numpy(m).reshape((1, 1, 80, 80)).to(device='cuda').float()).cpu().detach().numpy()

        a = np.argmax(a[0, :], axis=0)
        a = (a>0.5).astype(int)

        # b = matplotlib.image.imread(dir_mask_test_real+dataset_test.ids[j]+".png").convert('L')
        c = (np.asarray(Image.open(mouse_test_mask_path + dataset_test.ids[j] + ".png").convert('L')) /255 > .3).astype(int)

        if j == -1:
            plt.subplot(121)
            plt.imshow(a)
            plt.subplot(122)
            plt.imshow(np.abs(a-c))
            plt.show()
            plt.close()

        #plt.imshow(np.abs(c))
        #plt.show()

        sum += dice_score(gt=c, mask=a) # ((np.sum(np.abs(a-c)))/np.prod(np.shape(a)))

    print((sum/len(dataset_test.ids)))
    all_sum_mouse.append((sum/len(dataset_test.ids)))



mouse_test_img_path = "../../data/png/rat_test/img/"
mouse_test_mask_path = "../../data/png/rat_test/mask/"

amount = [str(x) for x in [5, 10, 20, 30, 40, 50]] + ["58", "64*"]
models = [f"mouse_{x}.pth" for x in amount[:-2]] + ["only_mouse.pth", "mouse_rat.pth"]

all_sum_rat = []
rat_residuuen = []
for i, model in enumerate(models):
    #net = UNet(n_channels=1, n_classes=2)

    #if i in [0, 1, 2, 3, 7]:
    net = UNet(n_channels=1, n_classes=2, bilinear=False)

    device = "cuda:0"
    net.to(device=device)
    net.load_state_dict(torch.load(model_path+model, map_location=device))

    dataset_test = BasicDataset(mouse_test_img_path, mouse_test_mask_path, 1)

    sum = 0
    for j in range(len(dataset_test.ids)):
        #a = net(dataset_test[j]["image"].reshape((1, 1, 80, 80)).to(device='cuda')).cpu().detach().numpy()

        m = np.asarray(Image.open(mouse_test_img_path + dataset_test.ids[j] + ".png").convert('L')) / 255
        a=net(torch.from_numpy(m).reshape((1, 1, 80, 80)).to(device='cuda').float()).cpu().detach().numpy()

        a = np.argmax(a[0, :], axis=0)
        a = (a>0.5).astype(int)

        # b = matplotlib.image.imread(dir_mask_test_real+dataset_test.ids[j]+".png").convert('L')
        c = (np.asarray(Image.open(mouse_test_mask_path + dataset_test.ids[j] + ".png").convert('L')) /255 > .3).astype(int)

        if j == 2:
            rat_residuuen.append(np.abs(a-c))

        sum += dice_score(gt=c, mask=a)  #((np.sum(np.abs(a-c)))/np.prod(np.shape(a)))
    print((sum/len(dataset_test.ids)))
    all_sum_rat.append((sum/len(dataset_test.ids)))

plt.close()
plt.plot([0, .5, 1.5, 2.5, 3.5, 4.5,  5.3, 5.9], np.array(all_sum_mouse), label="mouse accuracy")
plt.plot([0, .5, 1.5, 2.5, 3.5, 4.5,  5.3, 5.9], np.array(all_sum_rat), label="rat accuracy")
#plt.xticks(range(len(amount)), amount)
plt.xticks([0, .5, 1.5, 2.5, 3.5, 4.5, 5.3, 5.9], amount)
plt.ylabel("averaged dice-score")
plt.xlabel("quantity of animals")
plt.legend()

plt.savefig("../../demo/pixel_accuracy.png")


fig, axs = plt.subplots(1, 4, constrained_layout=True)
fig.set_size_inches(20, 6, forward=True)
fig.suptitle("Missclassifcation", fontsize=30)
axs[0].imshow(rat_residuuen[0], interpolation="nearest")
axs[0].set_xticks([], [])
axs[0].set_yticks([], [])
axs[0].set_xlabel("5 mice", fontsize=30)

axs[1].imshow(rat_residuuen[2], interpolation="nearest")
axs[1].set_xticks([], [])
axs[1].set_yticks([], [])
axs[1].set_xlabel("20 mice", fontsize=30)

axs[2].imshow(rat_residuuen[-2], interpolation="nearest")
axs[2].set_xticks([], [])
axs[2].set_yticks([], [])
axs[2].set_xlabel("58 mice", fontsize=30)

axs[3].imshow(rat_residuuen[-1], interpolation="nearest")
axs[3].set_xticks([], [])
axs[3].set_yticks([], [])
axs[3].set_xlabel("58 mice + 6 rats", fontsize=30)

plt.savefig("../../demo/missclassification.png")
