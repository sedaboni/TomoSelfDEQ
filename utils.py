import os
import numpy as np
from skimage import io
from datetime import datetime
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class ImageSaver():

    def __init__(self, root='results'):
        self.counter = 0
        # now = datetime.now()
        # now_string = now.strftime("%d_%m_%Y_%H:%M")

        self.path_save = f'{root}/images'

        if not os.path.exists(self.path_save):
            os.makedirs(self.path_save) 


    def __call__(self, xrecon, xtrue, epoch=0):

        psnr_list = []
        ssim_list = []
        
        for i, c in enumerate(xrecon):
            image_path = f'{self.path_save}/img_{self.counter}_{epoch}.jpg'
            image_rec = 255*c[0,...].detach().cpu().numpy()
            io.imsave(image_path, image_rec.astype(np.uint8))

            image_true = 255*xtrue[i, 0,...].detach().cpu().numpy()
            
            psnr_list.append(psnr(image_true.astype(np.uint8),image_rec.astype(np.uint8)))
            ssim_list.append(ssim(image_true.astype(np.uint8),image_rec.astype(np.uint8)))

            self.counter += 1
        
        return psnr_list, ssim_list

    def reset_counter(self):
        self.counter = 0
