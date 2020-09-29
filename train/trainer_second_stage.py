import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import dataset
import utils
import pwcnet
            
def trainer_noGAN(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    
    # Initialize Generator
    generator = utils.create_generator(opt)
    flownet = utils.create_pwcnet(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        flownet = flownet.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            model_name = 'Second_Stage_epoch%d_bs%d_%dp.pth' % (epoch, opt.batch_size, opt.crop_size_h)
        if opt.save_mode == 'iter':
            model_name = 'Second_Stage_iter%d_bs%d_%dp.pth' % (iteration, opt.batch_size, opt.crop_size_h)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines(opt.video_imagelist_txt)
    classlist = utils.text_readlines(opt.video_class_txt)

    # Define the dataset
    trainset = dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()

            loss_flow = 0
            loss_flow_short = 0
            loss_flow_long = 0
            loss_L1 = 0
            loss_percep = 0

            p_list = []                                                                         # save all the previous generated frames

            for iter_frame in range(opt.iter_frames):
                # Read data
                x_t = in_part[iter_frame].cuda()
                y_t = out_part[iter_frame].cuda()
                
                # Initialize the second input and compute flow from time t-1 => time t
                if iter_frame == 0:
                    p_t_last = in_part[0].cuda()
                else:
                    x_t_last = in_part[iter_frame - 1].cuda()                                   # range: [-1, 1]
                    p_t_last = p_t.detach()                                                     # range: [-1, 1]
                    p_t_last.requires_grad = False
                    p_list.append(p_t_last)                                                     # add the last generated frame to list
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)                   # range: [-20, +20]

                # Warp the last output to time t's place
                if iter_frame == 0:
                    p_t_last_warp = p_t_last
                else:
                    temp = p_t_last[:, [0], :, :] * 0.299 + p_t_last[:, [1], :, :] * 0.587 + p_t_last[:, [2], :, :] * 0.114
                    temp = torch.cat((temp, temp, temp), 1)
                    p_t_last_warp = pwcnet.PWCNetBackward((temp + 1) / 2, o_t_last_2_t)
                    p_t_last_warp = p_t_last_warp * 2 - 1

                # Generator output
                p_t = generator(x_t, p_t_last_warp)

                # Flow loss
                if iter_frame > 0:
                    o_p = pwcnet.PWCEstimate(flownet, p_t, p_t_last)
                    loss_flow += criterion_L1(o_p, o_t_last_2_t)

                # Short-term and Long-term loss
                if iter_frame > 0:
                    # Estimate the short-term loss
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)          # time t-1 => time t warp result; range: [0, 1]
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)          # range: [0, 1]
                    mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                    loss_flow_short += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                    # Estimate the long-term loss (dense)
                    if iter_frame > 1:
                        # the dense long-term loss loop
                        for iter_long_frame in range(iter_frame - 1):
                            # the following two lines extract the long range grayscale frame to compute the optical flow and mask; this 'last' is not last frame!!!
                            x_t_last = in_part[iter_long_frame].cuda()                          # this 'last' is not last frame; it is just to save memory!!!
                            o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)           # this 'last' is not last frame; it is just to save memory!!!
                            x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)  # time long range => time t warp result; range: [0, 1]
                            p_t_last = p_list[iter_long_frame]                                  # extract the long range generated frame to compute the warped reuslt; this 'last' is not last frame; it is just to save memory!!!
                            p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)  # range: [0, 1]
                            mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                            loss_flow_long += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                
                # Pixel-level loss
                loss_L1 += criterion_L1(p_t, y_t)

                # Perceptual Loss
                feature_fake_RGB = perceptualnet(p_t)
                feature_true_RGB = perceptualnet(y_t)
                loss_percep += criterion_L1(feature_fake_RGB, feature_true_RGB)

            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_flow * loss_flow + opt.lambda_flow_short * loss_flow_short + opt.lambda_flow_long * loss_flow_long + opt.lambda_percep * loss_percep
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            if opt.iter_frames > 2:
                # Print log
                print("[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.8f] [Flow Loss Short: %.8f] [Flow Loss Long: %.8f] Time_left: %s" %
                    ((epoch + opt.lr_overhead * 50 + 1), opt.epochs + opt.lr_overhead * 50 , iteration, len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(), loss_flow_long.item(), time_left))
            else:
                # Print log
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.8f] [Flow Loss Short: %.8f] [Flow Loss Long: %.8f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long, time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_G)
            
def trainer_LSGAN(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_MSE = torch.nn.MSELoss().cuda()
    
    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    flownet = utils.create_pwcnet(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        flownet = flownet.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            model_name = 'Second_Stage_epoch%d_bs%d_%dp.pth' % (epoch, opt.batch_size, opt.crop_size_h)
        if opt.save_mode == 'iter':
            model_name = 'Second_Stage_iter%d_bs%d_%dp.pth' % (iteration, opt.batch_size, opt.crop_size_h)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines(opt.video_imagelist_txt)
    classlist = utils.text_readlines(opt.video_class_txt)

    # Define the dataset
    trainset = dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            loss_flow = 0
            loss_flow_short = 0
            loss_flow_long = 0
            loss_L1 = 0
            loss_percep = 0
            loss_D = 0
            loss_G = 0

            # Adversarial ground truth
            valid = Tensor(np.ones((in_part[0].shape[0], 1, 30, 30)))
            fake = Tensor(np.zeros((in_part[0].shape[0], 1, 30, 30)))

            p_list = []                                                                         # save all the previous generated frames

            for iter_frame in range(opt.iter_frames):
                # Read data
                x_t = in_part[iter_frame].cuda()
                y_t = out_part[iter_frame].cuda()
                
                # Initialize the second input and compute flow from time t-1 => time t
                if iter_frame == 0:
                    p_t_last = in_part[0].cuda()
                else:
                    x_t_last = in_part[iter_frame - 1].cuda()                                   # range: [-1, 1]
                    p_t_last = p_t.detach()                                                     # range: [-1, 1]
                    p_t_last.requires_grad = False
                    p_list.append(p_t_last)                                                     # add the last generated frame to list
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)                   # range: [-20, +20]

                # Warp the last output to time t's place
                if iter_frame == 0:
                    p_t_last_warp = p_t_last
                else:
                    temp = p_t_last[:, [0], :, :] * 0.299 + p_t_last[:, [1], :, :] * 0.587 + p_t_last[:, [2], :, :] * 0.114
                    temp = torch.cat((temp, temp, temp), 1)
                    p_t_last_warp = pwcnet.PWCNetBackward((temp + 1) / 2, o_t_last_2_t)
                    p_t_last_warp = p_t_last_warp * 2 - 1

                # Generator output
                p_t = generator(x_t, p_t_last_warp)

                # Flow loss
                if iter_frame > 0:
                    o_p = pwcnet.PWCEstimate(flownet, p_t, p_t_last)
                    loss_flow += criterion_L1(o_p, o_t_last_2_t)

                # Short-term and Long-term loss
                if iter_frame > 0:
                    # Estimate the short-term loss
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)          # time t-1 => time t warp result; range: [0, 1]
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)          # range: [0, 1]
                    mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                    loss_flow_short += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                    # Estimate the long-term loss (dense)
                    if iter_frame > 1:
                        # the dense long-term loss loop
                        for iter_long_frame in range(iter_frame - 1):
                            # the following two lines extract the long range grayscale frame to compute the optical flow and mask; this 'last' is not last frame!!!
                            x_t_last = in_part[iter_long_frame].cuda()                          # this 'last' is not last frame; it is just to save memory!!!
                            o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)           # this 'last' is not last frame; it is just to save memory!!!
                            x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)  # time long range => time t warp result; range: [0, 1]
                            p_t_last = p_list[iter_long_frame]                                  # extract the long range generated frame to compute the warped reuslt; this 'last' is not last frame; it is just to save memory!!!
                            p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)  # range: [0, 1]
                            mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                            loss_flow_long += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                
                # Pixel-level loss
                loss_L1 += criterion_L1(p_t, y_t)

                # Perceptual Loss
                feature_fake_RGB = perceptualnet(p_t)
                feature_true_RGB = perceptualnet(y_t)
                loss_percep += criterion_L1(feature_fake_RGB, feature_true_RGB)

                # GAN Loss
                # Fake samples
                fake_scalar = discriminator(x_t, p_t.detach())
                loss_fake = criterion_MSE(fake_scalar, fake)
                # True samples
                true_scalar = discriminator(x_t, y_t)
                loss_true = criterion_MSE(true_scalar, valid)
                # Train Discriminator
                loss_D += 0.5 * (loss_fake + loss_true)
                # Train Generator
                fake_scalar = discriminator(x_t, p_t)
                loss_G += criterion_MSE(fake_scalar, valid)

            # Overall Loss and optimize
            loss = opt.lambda_l1 * loss_L1 + opt.lambda_flow * loss_flow + opt.lambda_flow_short * loss_flow_short + opt.lambda_flow_long * loss_flow_long + opt.lambda_percep * loss_percep + opt.lambda_gan * loss_G
            loss.backward()
            loss_D.backward()
            optimizer_G.step()
            optimizer_D.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            if opt.iter_frames > 2:
                # Print log
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long.item(), loss_G.item(), loss_D.item(), time_left))
            else:
                # Print log
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long, loss_G.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_D)
               
def trainer_WGAN(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    
    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    flownet = utils.create_pwcnet(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        flownet = flownet.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            model_name = 'Second_Stage_epoch%d_bs%d_%dp.pth' % (epoch, opt.batch_size, opt.crop_size_h)
        if opt.save_mode == 'iter':
            model_name = 'Second_Stage_iter%d_bs%d_%dp.pth' % (iteration, opt.batch_size, opt.crop_size_h)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines(opt.video_imagelist_txt)
    classlist = utils.text_readlines(opt.video_class_txt)

    # Define the dataset
    trainset = dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            loss_flow = 0
            loss_flow_short = 0
            loss_flow_long = 0
            loss_L1 = 0
            loss_percep = 0
            loss_D = 0
            loss_G = 0

            p_list = []                                                                         # save all the previous generated frames

            for iter_frame in range(opt.iter_frames):
                # Read data
                x_t = in_part[iter_frame].cuda()
                y_t = out_part[iter_frame].cuda()
                
                # Initialize the second input and compute flow from time t-1 => time t
                if iter_frame == 0:
                    p_t_last = in_part[0].cuda()
                else:
                    x_t_last = in_part[iter_frame - 1].cuda()                                   # range: [-1, 1]
                    p_t_last = p_t.detach()                                                     # range: [-1, 1]
                    p_t_last.requires_grad = False
                    p_list.append(p_t_last)                                                     # add the last generated frame to list
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)                   # range: [-20, +20]

                # Warp the last output to time t's place
                if iter_frame == 0:
                    p_t_last_warp = p_t_last
                else:
                    temp = p_t_last[:, [0], :, :] * 0.299 + p_t_last[:, [1], :, :] * 0.587 + p_t_last[:, [2], :, :] * 0.114
                    temp = torch.cat((temp, temp, temp), 1)
                    p_t_last_warp = pwcnet.PWCNetBackward((temp + 1) / 2, o_t_last_2_t)
                    p_t_last_warp = p_t_last_warp * 2 - 1

                # Generator output
                p_t = generator(x_t, p_t_last_warp)

                # Flow loss
                if iter_frame > 0:
                    o_p = pwcnet.PWCEstimate(flownet, p_t, p_t_last)
                    loss_flow += criterion_L1(o_p, o_t_last_2_t)

                # Short-term and Long-term loss
                if iter_frame > 0:
                    # Estimate the short-term loss
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)          # time t-1 => time t warp result; range: [0, 1]
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)          # range: [0, 1]
                    mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                    loss_flow_short += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                    # Estimate the long-term loss (dense)
                    if iter_frame > 1:
                        # the dense long-term loss loop
                        for iter_long_frame in range(iter_frame - 1):
                            # the following two lines extract the long range grayscale frame to compute the optical flow and mask; this 'last' is not last frame!!!
                            x_t_last = in_part[iter_long_frame].cuda()                          # this 'last' is not last frame; it is just to save memory!!!
                            o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)           # this 'last' is not last frame; it is just to save memory!!!
                            x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)  # time long range => time t warp result; range: [0, 1]
                            p_t_last = p_list[iter_long_frame]                                  # extract the long range generated frame to compute the warped reuslt; this 'last' is not last frame; it is just to save memory!!!
                            p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)  # range: [0, 1]
                            mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                            loss_flow_long += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                
                # Pixel-level loss
                loss_L1 += criterion_L1(p_t, y_t)

                # Perceptual Loss
                feature_fake_RGB = perceptualnet(p_t)
                feature_true_RGB = perceptualnet(y_t)
                loss_percep += criterion_L1(feature_fake_RGB, feature_true_RGB)

                # GAN Loss
                # Fake samples
                fake_scalar_d = discriminator(x_t, p_t.detach())
                # True samples
                true_scalar_d = discriminator(x_t, y_t)
                # Overall Loss and optimize
                loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d)
                # Train Generator
                fake_scalar = discriminator(x_t, p_t)
                loss_G = - torch.mean(fake_scalar)

            # Overall Loss and optimize
            loss = loss_L1 + opt.lambda_flow * loss_flow + opt.lambda_flow_short * loss_flow_short + opt.lambda_flow_long * loss_flow_long + opt.lambda_percep * loss_percep + opt.lambda_gan * loss_G
            loss.backward()
            loss_D.backward()
            optimizer_G.step()
            optimizer_D.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            if opt.iter_frames > 2:
                # Print log
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long.item(), loss_G.item(), loss_D.item(), time_left))
            else:
                # Print log
                print(
                    "\r%s: [Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long, loss_G.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_D)

def trainer_WGANGP(opt):

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
    if not os.path.exists(opt.sample_path):
        os.makedirs(opt.sample_path)

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()
    
    # Initialize Generator
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    flownet = utils.create_pwcnet(opt)
    perceptualnet = utils.create_perceptualnet(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
        discriminator = nn.DataParallel(discriminator)
        discriminator = discriminator.cuda()
        flownet = nn.DataParallel(flownet)
        flownet = flownet.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        discriminator = discriminator.cuda()
        generator = generator.cuda()
        flownet = flownet.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2))
    
    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, iteration, optimizer):
        #Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs
        if opt.lr_decrease_mode == 'epoch':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if opt.lr_decrease_mode == 'iter':
            lr = opt.lr_g * (opt.lr_decrease_factor ** (iteration // opt.lr_decrease_iter))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        if opt.save_mode == 'epoch':
            model_name = 'Second_Stage_epoch%d_bs%d_%dp.pth' % (epoch, opt.batch_size, opt.crop_size_h)
        if opt.save_mode == 'iter':
            model_name = 'Second_Stage_iter%d_bs%d_%dp.pth' % (iteration, opt.batch_size, opt.crop_size_h)
        save_name = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_name)
                    print('The trained model is saved as %s' % (model_name))
    
    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Define the class list
    imglist = utils.text_readlines(opt.video_imagelist_txt)
    classlist = utils.text_readlines(opt.video_class_txt)

    # Define the dataset
    trainset = dataset.MultiFramesDataset(opt, imglist, classlist)
    print('The overall number of classes:', len(trainset))

    # Define the dataloader
    dataloader = utils.create_dataloader(trainset, opt)
    
    # ----------------------------------------
    #                 Training
    # ----------------------------------------

    # Count start time
    prev_time = time.time()
    
    # Tensor type
    Tensor = torch.cuda.FloatTensor

    # Calculate the gradient penalty loss for WGAN-GP
    def compute_gradient_penalty(D, input_samples, real_samples, fake_samples):
    # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(input_samples, interpolates)
        # For PatchGAN
        fake = Variable(Tensor(real_samples.shape[0], 1, 30, 30).fill_(1.0), requires_grad = False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs = d_interpolates,
            inputs = interpolates,
            grad_outputs = fake,
            create_graph = True,
            retain_graph = True,
            only_inputs = True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
        return gradient_penalty

    # For loop training
    for epoch in range(opt.epochs):
        for iteration, (in_part, out_part) in enumerate(dataloader):
            
            # Train Generator
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            loss_flow = 0
            loss_flow_short = 0
            loss_flow_long = 0
            loss_L1 = 0
            loss_percep = 0
            loss_D = 0
            loss_G = 0

            p_list = []                                                                         # save all the previous generated frames

            for iter_frame in range(opt.iter_frames):
                # Read data
                x_t = in_part[iter_frame].cuda()
                y_t = out_part[iter_frame].cuda()
                
                # Initialize the second input and compute flow from time t-1 => time t
                if iter_frame == 0:
                    p_t_last = in_part[0].cuda()
                else:
                    x_t_last = in_part[iter_frame - 1].cuda()                                   # range: [-1, 1]
                    p_t_last = p_t.detach()                                                     # range: [-1, 1]
                    p_t_last.requires_grad = False
                    p_list.append(p_t_last)                                                     # add the last generated frame to list
                    o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)                   # range: [-20, +20]

                # Warp the last output to time t's place
                if iter_frame == 0:
                    p_t_last_warp = p_t_last
                else:
                    temp = p_t_last[:, [0], :, :] * 0.299 + p_t_last[:, [1], :, :] * 0.587 + p_t_last[:, [2], :, :] * 0.114
                    temp = torch.cat((temp, temp, temp), 1)
                    p_t_last_warp = pwcnet.PWCNetBackward((temp + 1) / 2, o_t_last_2_t)
                    p_t_last_warp = p_t_last_warp * 2 - 1

                # Generator output
                p_t = generator(x_t, p_t_last_warp)

                # Flow loss
                if iter_frame > 0:
                    o_p = pwcnet.PWCEstimate(flownet, p_t, p_t_last)
                    loss_flow += criterion_L1(o_p, o_t_last_2_t)

                # Short-term and Long-term loss
                if iter_frame > 0:
                    # Estimate the short-term loss
                    x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)          # time t-1 => time t warp result; range: [0, 1]
                    p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)          # range: [0, 1]
                    mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                    loss_flow_short += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                    # Estimate the long-term loss (dense)
                    if iter_frame > 1:
                        # the dense long-term loss loop
                        for iter_long_frame in range(iter_frame - 1):
                            # the following two lines extract the long range grayscale frame to compute the optical flow and mask; this 'last' is not last frame!!!
                            x_t_last = in_part[iter_long_frame].cuda()                          # this 'last' is not last frame; it is just to save memory!!!
                            o_t_last_2_t = pwcnet.PWCEstimate(flownet, x_t, x_t_last)           # this 'last' is not last frame; it is just to save memory!!!
                            x_t_warp = pwcnet.PWCNetBackward((x_t_last + 1) / 2, o_t_last_2_t)  # time long range => time t warp result; range: [0, 1]
                            p_t_last = p_list[iter_long_frame]                                  # extract the long range generated frame to compute the warped reuslt; this 'last' is not last frame; it is just to save memory!!!
                            p_t_warp = pwcnet.PWCNetBackward((p_t_last + 1) / 2, o_t_last_2_t)  # range: [0, 1]
                            mask_flow = torch.exp(- opt.mask_para * torch.sum((x_t + 1) / 2 - x_t_warp, dim = 1).pow(2)).unsqueeze(1)
                            loss_flow_long += criterion_L1(mask_flow * (p_t + 1) / 2, mask_flow * p_t_warp)
                
                # Pixel-level loss
                loss_L1 += criterion_L1(p_t, y_t)

                # Perceptual Loss
                feature_fake_RGB = perceptualnet(p_t)
                feature_true_RGB = perceptualnet(y_t)
                loss_percep += criterion_L1(feature_fake_RGB, feature_true_RGB)

                # GAN Loss
                # Fake samples
                fake_scalar_d = discriminator(x_t, p_t.detach())
                # True samples
                true_scalar_d = discriminator(x_t, y_t)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, x_t.data, y_t.data, p_t.data)
                # Overall Loss and optimize
                loss_D = - torch.mean(true_scalar_d) + torch.mean(fake_scalar_d) + opt.lambda_gp * gradient_penalty
                # Train Generator
                fake_scalar = discriminator(x_t, p_t)
                loss_G = - torch.mean(fake_scalar)

            # Overall Loss and optimize
            loss = loss_L1 + opt.lambda_flow * loss_flow + opt.lambda_flow_short * loss_flow_short + opt.lambda_flow_long * loss_flow_long + opt.lambda_percep * loss_percep + opt.lambda_gan * loss_G
            loss.backward()
            loss_D.backward()
            optimizer_G.step()
            optimizer_D.step()

            # Determine approximate time left
            iters_done = epoch * len(dataloader) + iteration
            iters_left = opt.epochs * len(dataloader) - iters_done
            time_left = datetime.timedelta(seconds = iters_left * (time.time() - prev_time))
            prev_time = time.time()

            if opt.iter_frames > 2:
                # Print log
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long.item(), loss_G.item(), loss_D.item(), time_left))
            else:
                # Print log
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [L1 Loss: %.4f] [Percep Loss: %.4f] [Flow loss: %.4f] [Flow Loss Short: %.4f] [Flow Loss Long: %.4f] [G Loss: %.4f] [D Loss: %.4f] Time_left: %s" %
                    ((epoch + opt.epoch_overhead + 1), opt.epochs, iteration,
                     len(dataloader), loss_L1.item(), loss_percep.item(), loss_flow.item(), loss_flow_short.item(),
                     loss_flow_long, loss_G.item(), loss_D.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), len(dataloader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_G)
            adjust_learning_rate(opt, (epoch + opt.epoch_overhead + 1), (iters_done + 1), optimizer_D)
        
        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [p_t, y_t]
            name_list = ['pred', 'gt']
            utils.save_sample_png(sample_folder = opt.sample_path, sample_name = 'epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list)
        