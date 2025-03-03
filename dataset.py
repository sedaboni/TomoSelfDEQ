import os
import torch
import cv2 as cv
import numpy as np
import LION.CTtools.ct_utils as ct
from torch.utils.data import Dataset
from skimage.transform import  resize
import LION.CTtools.ct_geometry as ctgeo 

def get_images(path):
    all_images = []
    all_image_names = os.listdir(path)
    for name in all_image_names:
        temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
        image = temp_image[90:410, 90:410]
        image = image[0:320, 0:320]
        image = image / 0.07584485627272729
        all_images.append(image)

    return all_images

def get_dataset(path="data/walnuts"):
    images = get_images(path)
    images = np.array(images, dtype='float32')
    Images = np.zeros((37,456,456))
    Images[:,68:-68,68:-68]=images
    Images = resize(Images, (37,336,336))
    images = Images[5:]
    return images

def parallel_default_parameters(image_shape, number_of_angles):
    return ctgeo.Geometry(
                image_shape=image_shape,
                image_size=image_shape,
                detector_shape=image_shape[0:2],
                detector_size=image_shape[0:2],
                dso=image_shape[1] * 2,
                dsd=image_shape[1] * 4,
                mode="parallel",
                angles=np.linspace(0, np.pi, number_of_angles, endpoint=False))

class TrainWalnuts(Dataset):

    def __init__(self, nangles, noise_level):
        super().__init__()

        self.dev = torch.device("cuda")
        self.images = get_dataset()
        self.image_shape = self.images[0].shape

        self.geo = parallel_default_parameters(image_shape=[1,self.image_shape[0],self.image_shape[1]],number_of_angles=384)  # parallel beam standard CT
        self.op = ct.make_operator(self.geo)

        self.full_angles = np.arange(384)
        self.nangles = nangles

        self.noise_level = noise_level

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        im = torch.tensor(self.images[idx]).unsqueeze(0)
        sino = self.op(im)

        mask_temp = np.random.choice(self.full_angles, self.nangles, replace=False)
        mask_temp = np.sort(mask_temp)
        mask = torch.zeros_like(sino)
        mask[:,mask_temp,:] = 1.
        sino_res = torch.zeros_like(sino)
        sino_res[:,mask_temp,:] = sino[:,mask_temp,:] + self.noise_level*torch.max(sino)*torch.randn(sino[:,mask_temp,:].size())

        mask_temp2 = np.random.choice(self.full_angles, self.nangles, replace=False)
        mask_temp2 = np.sort(mask_temp2)
        mask2 = torch.zeros_like(sino)
        mask2[:,mask_temp2,:] = 1.
        sino_res2 = torch.zeros_like(sino)
        sino_res2[:,mask_temp2,:] = sino[:,mask_temp2,:] + self.noise_level*torch.max(sino)*torch.randn(sino[:,mask_temp2,:].size())

        return sino_res.to(self.dev), im.to(self.dev), mask.to(self.dev), sino_res2.to(self.dev), im.to(self.dev), mask2.to(self.dev)
    
class TestWalnuts(Dataset):

    def __init__(self, nangles, noise_level):
        super().__init__()

        self.dev = torch.device("cuda")
        self.images = get_dataset()
        self.image_shape = self.images[0].shape

        self.nangles = nangles
        self.geo = parallel_default_parameters(image_shape=[1,self.image_shape[0],self.image_shape[1]],number_of_angles=384)  # parallel beam standard CT
        self.op = ct.make_operator(self.geo)

        self.noise_level = noise_level
        self.noise_tensor = torch.randn(len(self.images), 384,336)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        
        im = torch.tensor(self.images[idx]).unsqueeze(0)
        sino = self.op(im)
        mask_temp = np.arange(0,384,384//self.nangles)
        mask = torch.zeros_like(sino)
        mask[:,mask_temp,:] = 1.
        sino_res = torch.zeros_like(sino)
        sino_res[:,mask_temp,:] = sino[:,mask_temp,:] + self.noise_level*torch.max(sino)*self.noise_tensor[idx,mask_temp,:]

        return sino_res.cuda(), im.cuda(), mask.cuda()