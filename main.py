import os
import torch
import random
import argparse
import numpy as np
from deq_model import DEQModel
from utils import ImageSaver
from torch.utils.data import DataLoader
from dataset import TrainWalnuts, TestWalnuts
from torch.utils.tensorboard import SummaryWriter
from adamw_schedulefree import AdamWScheduleFree

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def main():
    # Model settings
    parser = argparse.ArgumentParser(description='Model example')
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--N_epochs', type=int, default=2000)
    parser.add_argument('--nangles', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--noise_level', type=float, default=1e-2)
    parser.add_argument('--comment', type=str, default="")
    parser.add_argument('--loss_type', type=str, default='unsup')

    args = parser.parse_args()

    log_dir = os.path.join("runs", f'{args.nangles}_{args.loss_type}_{args.comment}')
    saver = ImageSaver(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    mse = torch.nn.MSELoss()

    dataset_test = TestWalnuts(args.nangles, args.noise_level)
    test_loader = DataLoader(dataset_test, batch_size=8)
    dataset_train = TrainWalnuts(args.nangles, args.noise_level)
    train_loader = DataLoader(dataset_train, batch_size=8)

    net = DEQModel(op = dataset_train.op, nangles=args.nangles).cuda()
    optimizer = AdamWScheduleFree(net.parameters(), lr=args.lr)

    for epoch in range(args.N_epochs):

        net.train()
        optimizer.train()
        
        train_losses = []
        for sinos, ims, masks, sinos2, ims, masks2 in train_loader:
            optimizer.zero_grad()

            x0 = torch.zeros_like(ims)
            xhat = net(x0, sinos, masks)
            
            # unsup loss
            if args.loss_type=='unsup':
                A_xhat = masks2*net.block.A(xhat)
                loss = np.sqrt(6*(384/args.nangles))*np.sqrt(net.block.scaling)*mse(A_xhat, sinos2)

            # derived loss
            elif args.loss_type=='derived':
                loss = np.sqrt(6*(384/args.nangles))*np.sqrt(net.block.scaling)*mse(net.block.A(xhat), net.block.A(ims))

            # usual loss
            elif args.loss_type=='usual':
                loss = mse(xhat, ims)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()

            train_losses.append(loss.item())
        
            
        writer.add_scalar('Loss/train', np.mean(train_losses), epoch)
        print(f'Done batch epoch {epoch}')


        if epoch%5 == 0:
            test_losses = []

            psnr_list = []
            ssim_list = []
            
            net.eval()
            optimizer.eval()

            for sinos, ims, masks in test_loader:
                x0 = torch.zeros_like(ims)
                xhat = net(x0, sinos, masks)
                test_loss = net.block.scaling*mse(net.block.A(xhat), net.block.A(ims))
                test_losses.append(test_loss.item())

                # temp_psnr_list, temp_ssim_list = saver(xhat, ims, epoch)
                temp_psnr_list, temp_ssim_list = saver(xhat, ims)

                psnr_list += temp_psnr_list
                ssim_list += temp_ssim_list

            writer.add_scalar('Loss/test', np.mean(test_losses), epoch)
            writer.add_scalar('Metrics/PSNR', np.mean(psnr_list), epoch)
            writer.add_scalar('Metrics/SSIM', np.mean(ssim_list), epoch)
            writer.add_scalar('log/min', torch.min(xhat).item(), epoch)
            writer.add_scalar('log/max', torch.max(xhat).item(), epoch)
            saver.reset_counter()
        


if __name__ == '__main__':
    main()