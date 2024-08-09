#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import time
import os
import torch
import torch.nn.functional as F
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
from gaussian_renderer import GaussianRasterizationSettings, GaussianRasterizer
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from decoder.init_decoder import DecoderAttributes
from pytorch_wavelets import DWTInverse, DWTForward
from torchviz import make_dot
import math
import matplotlib.pyplot as plt
from utils.graphics_utils import getWorld2View
from utils.voxelization import voxelization
import open3d as o3d
from lpipsPyTorch import lpips
import wandb

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


'''
rendered_old : old full-resolution image which is rendered first.
rendered_new : new full-resolution image which is rendered next. (for the patchwise one)
GT_image : the ground truth image. (original)

computes gradients for the full resolution and the patchwise resolution

returns nothing

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs

def compute_left_up(z, fovx, fovy):
    left = z*math.tan(fovx/2)
    up = z*math.tan(fovy/2)
    return left, up

def training(dataset, opt, pipe, testing_iterations, saving_iterations, saving_epochs, checkpoint_iterations, checkpoint, debug_from, device=device):
    
    # wandb.init(
    # # set the wandb project where this run will be logged
    # project="epoch_100_blender_{}".format(args.data_name))

    first_epoch = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    # # mask 생성
    # gaussians.create_mask() 
    # print('======================================================')  
    # print(gaussians._xyz.shape)
    scene = Scene(dataset, gaussians, load_iteration=-1)
    gaussians.training_setup(opt) 
    # mask 생성
    # gaussians.create_mask()
    # gaussians.mask_to_optimizer( gaussians._mask)
    # print(gaussians._mask)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    print(f'\n>>> Loading decoder from {args.msg_decoder_path}...')
    dec_attrs = DecoderAttributes('./decoder/cfg.json')

    msg_decoder = dec_attrs.dec
    msg_decoder.eval()
    # hyperparams
    lambda_i = dec_attrs.lambda_i
    lambda_l1 = dec_attrs.lambda_l1
    lambda_ssim = dec_attrs.lambda_ssim
    lambda_tv = dec_attrs.lambda_tv

    loss_type = dec_attrs.loss_dict
    
    # target message
    gt_msg = dec_attrs.msg
    # patch size
    patch_size = dec_attrs.patch_size

    ema_loss_for_log = 0.0
    first_epoch += 1

    gstep = 0
    progress_bar = tqdm(range(0, args.epochs), total = args.epochs)

    torch.cuda.empty_cache()
    
    
    for epoch_id in progress_bar: 
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        num_iteration =len(viewpoint_stack)
        print(f'number of cameras : {num_iteration}')

        log_loss = 0.0
        log_psnr = 0.0
        log_bit_acc = 0.0
        log_ssim = 0.0
        log_lpips = 0.0

        for iter_id in range(len(viewpoint_stack)):    
            gstep += 1
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            bg = torch.rand((3), device=device) if opt.random_background else background
        
            # with torch.no_grad(): 
            
            fovy, fovx = viewpoint_cam.FoVy, viewpoint_cam.FoVx
            # make rasterization settings to find points in the view frustum
            raster_settings = GaussianRasterizationSettings(
                image_height=int(viewpoint_cam.image_height),
                image_width=int(viewpoint_cam.image_width),
                tanfovx=fovx/2,
                tanfovy=fovy/2, 
                bg=bg_color,
                scale_modifier=1.0,
                viewmatrix=viewpoint_cam.world_view_transform,
                projmatrix=viewpoint_cam.full_proj_transform,
                sh_degree=gaussians.active_sh_degree,
                campos=viewpoint_cam.camera_center,
                prefiltered=False,
                debug=pipe.debug
            )
 
            # 이 부분이 30초~60초 정도 소요 -> iteration이 진행될수록 오래 걸림(가우시안이 추가되어서)
            xyz_in_frustum_mask = GaussianRasterizer(raster_settings).markVisible(gaussians._xyz)
            xyz_in_frustum = gaussians._xyz[xyz_in_frustum_mask] 
            sparse_xyz_in_frustum, opacities, feat_dcs, feat_rests = voxelization(xyz_in_frustum, voxel_size=5, gaussians=gaussians)
            rand_gaussians = gaussians.generate_gaussians(pos=sparse_xyz_in_frustum, 
                                                              opacity=opacities,
                                                              color= (feat_dcs, feat_rests),
                                                              min_scale= True)
      
            # print("before postfix mask shape : ", gaussians._mask.shape)
            # print(rand_gaussians['mask'])
            # combain parameters with existing parameters
            gaussians.densification_postfix(new_xyz=rand_gaussians['xyz'],
                                            new_features_dc=rand_gaussians['f_dc'],
                                            new_features_rest=rand_gaussians['f_rest'],
                                            new_opacities=rand_gaussians['opacity'],
                                            new_scaling=rand_gaussians['scaling'],
                                            new_rotation=rand_gaussians['rotation']
                                            ) #new_mask=rand_gaussians['mask'])

            # print("after postfix mask shape : ", gaussians._mask.shape)
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            
            
            gt_image = viewpoint_cam.original_image.cuda()

            image = image.unsqueeze(0).contiguous()
            gt_image = gt_image.unsqueeze(0).contiguous().to('cuda')
            # print("image shape : ", image.shape)            

            # image.requires_grad_(True)
            LL_img, _ = DWTForward(wave='bior4.4', J=1, mode='periodization').to(device)(image)
            # Extract watermark
            decoded = msg_decoder(LL_img) # b c h w -> b k

            loss_wm = loss_type['loss_w'](decoded, gt_msg)
            # loss_im = loss_type['loss_i'](image, gt_image)

            loss_im_mse = F.mse_loss(gt_image, image)
            psnr = -10.0 * math.log10(loss_im_mse)

            Ll1 = l1_loss(image, gt_image)
            loss = (0.8 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.2 * loss_wm  #+ opt.lambda_mask*torch.mean((torch.sigmoid(gaussians._mask)))
            # total_loss = lambda_i * loss_im + (1-lambda_i) * loss_wm

            loss_dict = {}
            loss_dict['watermark_loss'] = loss_wm.detach().item()
            loss_dict['image_loss'] = Ll1.detach().item()
            
            loss_dict['psnr'] = psnr
            loss_dict['ssim'] = ssim(image.squeeze(0).permute(1,2,0), gt_image.squeeze(0).permute(1,2,0)).item()
            loss_dict['lpips'] = lpips(image, gt_image, net_type='vgg').item()
        
            loss_dict['bit-accuracy'] = bit_acc(decoded, gt_msg).item()
            loss_dict['total_loss'] = loss.detach().item()
                        
            loss.backward()
            
            log_loss += loss_dict['total_loss']
            log_psnr += loss_dict['psnr']
            log_bit_acc += loss_dict['bit-accuracy']
            log_ssim += loss_dict['ssim']
            log_lpips += loss_dict['lpips']

            torch.cuda.empty_cache()

            print()
            progress_bar.set_description(f'global-iteration-{gstep} | epoch-{epoch_id+1} : loss_im={loss_dict["image_loss"]:.4f} \
                                 loss_wm={loss_dict["watermark_loss"]:.4f} bit-accuracy={loss_dict["bit-accuracy"]:.4f} psnr={loss_dict["psnr"]:.4f} \
                                 ssim={loss_dict["ssim"]:.4f} lpips={loss_dict["lpips"]:.4f}')
            
            

            gaussians.update_learning_rate(gstep)
            
            ####################################################################################################
            with torch.no_grad():

                # # destification
                # if gstep < opt.densify_until_iter:
                #     # print('Before destification gaussian_numbers : ', gaussians._xyz.shape)
                #     # Keep track of max radii in image-space for pruning
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if gstep > opt.densify_from_iter and gstep % opt.densification_interval == 0:
                #             size_threshold = 20 if gstep > opt.opacity_reset_interval else None
                #             gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                #             # print('destification')
                #     # if gstep % opt.opacity_reset_interval == 0 or (dataset.white_background and gstep == opt.densify_from_iter):
                #     #     gaussians.reset_opacity()
                #     # print('After destification gaussian_numbers : ', gaussians._xyz.shape)
                # else:
                #     if gstep % opt.mask_prune_iter == 0:
                #         gaussians.mask_prune()

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

        # wandb.log({"Training loss": log_loss / num_iteration},step= epoch_id)
        # wandb.log({"Training psnr": log_psnr / num_iteration},step= epoch_id)
        # wandb.log({"Training ssim": log_ssim / num_iteration},step= epoch_id)
        # wandb.log({"Training bit acc": log_loss / num_iteration},step= epoch_id)

        tb_writer.add_scalar("Loss/train", log_loss / num_iteration, epoch_id)
        tb_writer.add_scalar("PSNR/train", log_psnr / num_iteration, epoch_id)
        tb_writer.add_scalar("SSIM/train", log_ssim / num_iteration, epoch_id)
        tb_writer.add_scalar("LPIPS/train", log_lpips / num_iteration, epoch_id)
        tb_writer.add_scalar("Bit_acc/train", log_bit_acc / num_iteration, epoch_id)

        with torch.no_grad():
            if (epoch_id+1 in saving_epochs):
                print("\n[ITER {}] Saving Gaussians".format(epoch_id+1))
                scene.save(epoch_id+1)
                
            loss_dict_test = {"loss":0.0, "psnr":0.0, "ssim":0.0, "lpips":0.0,"bit-accuracy":0.0, "GT bit-accuracy (It must be low)": 0.0}
            # Progress bar
            test_viewpoint_stack = scene.getTestCameras()
            test_viewpoint_cam = test_viewpoint_stack
            for idx, view in enumerate(test_viewpoint_cam[:3]):
                test_renderd_image = render(view, gaussians, pipe, bg)["render"]
                test_gt = view.original_image[0:3, :, :]
                test_image = test_renderd_image.unsqueeze(0).contiguous().to('cuda')
                test_gt_image = test_gt.unsqueeze(0).contiguous().to('cuda')
                tb_writer.add_image(f'test_image/pred_wm_{idx:04d}', test_image.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=epoch_id, dataformats='HWC')
                tb_writer.add_image(f'test_image/gt_{idx:04d}', test_gt_image.squeeze(0).permute(1,2,0).cpu().clamp_max_(1.0), global_step=epoch_id, dataformats='HWC')
                # wandb.log({"rendered image {idx}" : wandb.Image(test_image)})
                # wandb.log({"gt image {idx}" : wandb.Image(test_gt_image)})

                LL_img, _ = DWTForward(wave='bior4.4', J=1, mode='periodization').to(device)(test_image)
                no_LL_img, _ = DWTForward(wave='bior4.4', J=1, mode='periodization').to(device)(test_gt_image)
                decoded = msg_decoder(LL_img)

                loss_im_mse = F.mse_loss(test_gt_image, test_image)
                psnr = -10.0 * math.log10(loss_im_mse)
                loss_wm_test = loss_type['loss_w'](decoded, gt_msg)
                test_Ll1 = l1_loss(test_image, test_gt_image)
                test_loss = (0.9 - opt.lambda_dssim) * test_Ll1 + opt.lambda_dssim * (1.0 - ssim(test_image, test_gt_image)) + 0.1 * loss_wm_test  + opt.lambda_mask*torch.mean((torch.sigmoid(gaussians._mask)))

                loss_dict_test['loss'] += test_loss
                loss_dict_test['psnr'] += psnr
                loss_dict_test['ssim'] += ssim(test_image.squeeze(0).permute(1,2,0), test_gt_image.squeeze(0).permute(1,2,0)).item()
                loss_dict_test['lpips'] += lpips(test_image, test_gt_image, net_type='vgg').item()
                loss_dict_test['bit-accuracy'] += bit_acc(decoded, gt_msg).item()
                loss_dict_test["GT bit-accuracy (It must be low)"] += bit_acc(msg_decoder(no_LL_img), gt_msg).item()

            for stat_name in loss_dict_test:
                loss_dict_test[stat_name] /= len(test_viewpoint_cam[:3])
            print('<validation stats>\n')
            for stat_name in loss_dict_test:
                print(f'{stat_name} : {loss_dict_test[stat_name]:.4f}')
            print()

            tb_writer.add_scalar("Loss/test", loss_dict_test['loss'], epoch_id)
            tb_writer.add_scalar("PSNR/test", loss_dict_test['psnr'], epoch_id)
            tb_writer.add_scalar("SSIM/test", loss_dict_test['ssim'], epoch_id)
            tb_writer.add_scalar("LPIPS/test", loss_dict_test['lpips'], epoch_id)
            tb_writer.add_scalar("Bit_acc/test", loss_dict_test['bit-accuracy'], epoch_id)


            # wandb.log({"Test loss": loss_dict_test['loss']},step= epoch_id)
            # wandb.log({"Test psnr": loss_dict_test['psnr']},step= epoch_id)
            # wandb.log({"Test ssim": loss_dict_test['ssim']},step= epoch_id)
            # wandb.log({"Test bit acc": loss_dict_test['bit-accuracy']},step= epoch_id)

    # wandb.finish()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(device), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_epochs", nargs="+", type=int, default=[1, 2, 3, 5, 10, 20,30,40,50,60,70,80,90, 100,200,300])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--msg_decoder_path", type=str, default = "/home/jang/gaussian-splatting/decoder/pretrained_decoder/16bits/16_256_checkpoint_whit.pth")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.save_epochs, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")