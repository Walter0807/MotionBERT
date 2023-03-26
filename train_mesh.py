import os
import random
import copy
import time
import sys
import shutil
import argparse
import errno
import math
import numpy as np
from collections import defaultdict, OrderedDict
import tensorboardX
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from lib.utils.tools import *
from lib.model.loss import *
from lib.model.loss_mesh import *
from lib.utils.utils_mesh import *
from lib.utils.utils_smpl import *
from lib.utils.utils_data import *
from lib.utils.learning import *
from lib.data.dataset_mesh import MotionSMPL
from lib.model.model_mesh import MeshRegressor
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def validate(test_loader, model, criterion, dataset_name='h36m'):
    model.eval()
    print(f'===========> validating {dataset_name}')
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_dict = {'loss_3d_pos': AverageMeter(), 
                   'loss_3d_scale': AverageMeter(), 
                   'loss_3d_velocity': AverageMeter(),
                   'loss_lv': AverageMeter(), 
                   'loss_lg': AverageMeter(), 
                   'loss_a': AverageMeter(), 
                   'loss_av': AverageMeter(), 
                   'loss_pose': AverageMeter(), 
                   'loss_shape': AverageMeter(),
                   'loss_norm': AverageMeter(),
    }
    mpjpes = AverageMeter()
    mpves = AverageMeter()
    results = defaultdict(list)
    smpl = SMPL(args.data_root, batch_size=1).cuda()
    J_regressor = smpl.J_regressor_h36m
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size, clip_len = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_gt['theta'] = batch_gt['theta'].cuda().float()
                batch_gt['kp_3d'] = batch_gt['kp_3d'].cuda().float()
                batch_gt['verts'] = batch_gt['verts'].cuda().float()
                batch_input = batch_input.cuda().float()
            output = model(batch_input)    
            output_final = output
            if args.flip:
                batch_input_flip = flip_data(batch_input)
                output_flip = model(batch_input_flip)
                output_flip_pose = output_flip[0]['theta'][:, :, :72]
                output_flip_shape = output_flip[0]['theta'][:, :, 72:]
                output_flip_pose = flip_thetas_batch(output_flip_pose)
                output_flip_pose = output_flip_pose.reshape(-1, 72)
                output_flip_shape = output_flip_shape.reshape(-1, 10)
                output_flip_smpl = smpl(
                    betas=output_flip_shape,
                    body_pose=output_flip_pose[:, 3:],
                    global_orient=output_flip_pose[:, :3],
                    pose2rot=True
                )
                output_flip_verts = output_flip_smpl.vertices.detach()*1000.0
                J_regressor_batch = J_regressor[None, :].expand(output_flip_verts.shape[0], -1, -1).to(output_flip_verts.device)
                output_flip_kp3d = torch.matmul(J_regressor_batch, output_flip_verts)  # (NT,17,3) 
                output_flip_back = [{
                    'theta': torch.cat((output_flip_pose.reshape(batch_size, clip_len, -1), output_flip_shape.reshape(batch_size, clip_len, -1)), dim=-1),
                    'verts': output_flip_verts.reshape(batch_size, clip_len, -1, 3),
                    'kp_3d': output_flip_kp3d.reshape(batch_size, clip_len, -1, 3),
                }]
                output_final = [{}]
                for k, v in output_flip[0].items():
                    output_final[0][k] = (output[0][k] + output_flip_back[0][k])*0.5
                output = output_final
            loss_dict = criterion(output, batch_gt)
            loss = args.lambda_3d      * loss_dict['loss_3d_pos']      + \
                   args.lambda_scale   * loss_dict['loss_3d_scale']    + \
                   args.lambda_3dv     * loss_dict['loss_3d_velocity'] + \
                   args.lambda_lv      * loss_dict['loss_lv']          + \
                   args.lambda_lg      * loss_dict['loss_lg']          + \
                   args.lambda_a       * loss_dict['loss_a']           + \
                   args.lambda_av      * loss_dict['loss_av']          + \
                   args.lambda_shape   * loss_dict['loss_shape']       + \
                   args.lambda_pose    * loss_dict['loss_pose']        + \
                   args.lambda_norm    * loss_dict['loss_norm'] 
            # update metric
            losses.update(loss.item(), batch_size)
            loss_str = ''
            for k, v in loss_dict.items():
                losses_dict[k].update(v.item(), batch_size)
                loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])
            mpjpe, mpve = compute_error(output, batch_gt)
            mpjpes.update(mpjpe, batch_size)
            mpves.update(mpve, batch_size)
            
            for keys in output[0].keys():
                output[0][keys] = output[0][keys].detach().cpu().numpy()
                batch_gt[keys] = batch_gt[keys].detach().cpu().numpy()
            results['kp_3d'].append(output[0]['kp_3d'])
            results['verts'].append(output[0]['verts'])
            results['kp_3d_gt'].append(batch_gt['kp_3d'])
            results['verts_gt'].append(batch_gt['verts'])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % int(opts.print_freq) == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      '{2}'
                      'PVE {mpves.val:.3f} ({mpves.avg:.3f})\t'
                      'JPE {mpjpes.val:.3f} ({mpjpes.avg:.3f})'.format(
                       idx, len(test_loader), loss_str, batch_time=batch_time,
                       loss=losses, mpves=mpves, mpjpes=mpjpes))

    print(f'==> start concating results of {dataset_name}')
    for term in results.keys():
        results[term] = np.concatenate(results[term])
    print(f'==> start evaluating {dataset_name}...')
    error_dict = evaluate_mesh(results)
    err_str = ''
    for err_key, err_val in error_dict.items():
        err_str += '{}: {:.2f}mm \t'.format(err_key, err_val)
    print(f'=======================> {dataset_name} validation done: ', loss_str)
    print(f'=======================> {dataset_name} validation done: ', err_str)
    return losses.avg, error_dict['mpjpe'], error_dict['pa_mpjpe'], error_dict['mpve'], losses_dict


def train_epoch(args, opts, model, train_loader, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch):
    model.train()
    end = time.time()
    for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - end)
        batch_size = len(batch_input)

        if torch.cuda.is_available():
            batch_gt['theta'] = batch_gt['theta'].cuda().float()
            batch_gt['kp_3d'] = batch_gt['kp_3d'].cuda().float()
            batch_gt['verts'] = batch_gt['verts'].cuda().float()
            batch_input = batch_input.cuda().float()
        output = model(batch_input)
        optimizer.zero_grad()
        loss_dict = criterion(output, batch_gt)
        loss_train = args.lambda_3d      * loss_dict['loss_3d_pos']      + \
                     args.lambda_scale   * loss_dict['loss_3d_scale']    + \
                     args.lambda_3dv     * loss_dict['loss_3d_velocity'] + \
                     args.lambda_lv      * loss_dict['loss_lv']          + \
                     args.lambda_lg      * loss_dict['loss_lg']          + \
                     args.lambda_a       * loss_dict['loss_a']           + \
                     args.lambda_av      * loss_dict['loss_av']          + \
                     args.lambda_shape   * loss_dict['loss_shape']       + \
                     args.lambda_pose    * loss_dict['loss_pose']        + \
                     args.lambda_norm    * loss_dict['loss_norm'] 
        losses_train.update(loss_train.item(), batch_size)
        loss_str = ''
        for k, v in loss_dict.items():
            losses_dict[k].update(v.item(), batch_size)
            loss_str += '{0} {loss.val:.3f} ({loss.avg:.3f})\t'.format(k, loss=losses_dict[k])
        
        mpjpe, mpve = compute_error(output, batch_gt)
        mpjpes.update(mpjpe, batch_size)
        mpves.update(mpve, batch_size)
        
        loss_train.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if idx % int(opts.print_freq) == 0:
            print('Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                '{3}'
                'PVE {mpves.val:.3f} ({mpves.avg:.3f})\t'
                'JPE {mpjpes.val:.3f} ({mpjpes.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), loss_str, batch_time=batch_time,
                data_time=data_time, loss=losses_train, mpves=mpves, mpjpes=mpjpes))
            sys.stdout.flush()

def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
        shutil.copy(opts.config, opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_pos']
            model_backbone = load_pretrained_weights(model_backbone, checkpoint)
    if args.partial_train:
        model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = MeshRegressor(args, backbone=model_backbone, dim_rep=args.dim_rep, hidden_dim=args.hidden_dim, dropout_ratio=args.dropout, num_joints=args.num_joints)
    criterion = MeshLoss(loss_type = args.loss_type)
    best_jpe = 9999.0
    model_params = 0
    for parameter in model.parameters():
        if parameter.requires_grad == True:
            model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 8,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    if hasattr(args, "dt_file_h36m"):
        mesh_train = MotionSMPL(args, data_split='train', dataset="h36m")
        mesh_val = MotionSMPL(args, data_split='test', dataset="h36m")
        train_loader = DataLoader(mesh_train, **trainloader_params)
        test_loader = DataLoader(mesh_val, **testloader_params)
        print('INFO: Training on {} batches (h36m)'.format(len(train_loader)))

    if hasattr(args, "dt_file_pw3d"):
        if args.train_pw3d:
            mesh_train_pw3d = MotionSMPL(args, data_split='train', dataset="pw3d")
            train_loader_pw3d = DataLoader(mesh_train_pw3d, **trainloader_params)
            print('INFO: Training on {} batches (pw3d)'.format(len(train_loader_pw3d)))
        mesh_val_pw3d = MotionSMPL(args, data_split='test', dataset="pw3d")
        test_loader_pw3d = DataLoader(mesh_val_pw3d, **testloader_params)
    
    
    trainloader_img_params = {
            'batch_size': args.batch_size_img,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
        }
    testloader_img_params = {
            'batch_size': args.batch_size_img,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
        }
    
    if hasattr(args, "dt_file_coco"):
        mesh_train_coco = MotionSMPL(args, data_split='train', dataset="coco")
        mesh_val_coco = MotionSMPL(args, data_split='test', dataset="coco")
        train_loader_coco = DataLoader(mesh_train_coco, **trainloader_img_params)
        test_loader_coco = DataLoader(mesh_val_coco, **testloader_img_params)
        print('INFO: Training on {} batches (coco)'.format(len(train_loader_coco)))

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
    if not opts.evaluate:
        optimizer = optim.AdamW(
                [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                      {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
                ],      lr=args.lr_backbone, 
                        weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
            lr = checkpoint['lr']
            if 'best_jpe' in checkpoint and checkpoint['best_jpe'] is not None:
                best_jpe = checkpoint['best_jpe']
        
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            losses_dict = {
                'loss_3d_pos': AverageMeter(), 
                'loss_3d_scale': AverageMeter(), 
                'loss_3d_velocity': AverageMeter(),
                'loss_lv': AverageMeter(), 
                'loss_lg': AverageMeter(), 
                'loss_a': AverageMeter(), 
                'loss_av': AverageMeter(), 
                'loss_pose': AverageMeter(), 
                'loss_shape': AverageMeter(),
                'loss_norm': AverageMeter(),
            }
            mpjpes = AverageMeter()
            mpves = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            
            if hasattr(args, "dt_file_h36m") and epoch < args.warmup_h36m:
                train_epoch(args, opts, model, train_loader, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)
                test_loss, test_mpjpe, test_pa_mpjpe, test_mpve, test_losses_dict = validate(test_loader, model, criterion, 'h36m')
                for k, v in test_losses_dict.items():
                    train_writer.add_scalar('test_loss/'+k, v.avg, epoch + 1)
                train_writer.add_scalar('test_loss', test_loss, epoch + 1)
                train_writer.add_scalar('test_mpjpe', test_mpjpe, epoch + 1)
                train_writer.add_scalar('test_pa_mpjpe', test_pa_mpjpe, epoch + 1)
                train_writer.add_scalar('test_mpve', test_mpve, epoch + 1)

            if hasattr(args, "dt_file_coco") and epoch < args.warmup_coco:
                train_epoch(args, opts, model, train_loader_coco, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)
            
            if hasattr(args, "dt_file_pw3d"):
                if args.train_pw3d:
                    train_epoch(args, opts, model, train_loader_pw3d, losses_train, losses_dict, mpjpes, mpves, criterion, optimizer, batch_time, data_time, epoch)    
                test_loss_pw3d, test_mpjpe_pw3d, test_pa_mpjpe_pw3d, test_mpve_pw3d, test_losses_dict_pw3d = validate(test_loader_pw3d, model, criterion, 'pw3d')  
                for k, v in test_losses_dict_pw3d.items():
                    train_writer.add_scalar('test_loss_pw3d/'+k, v.avg, epoch + 1)
                train_writer.add_scalar('test_loss_pw3d', test_loss_pw3d, epoch + 1)
                train_writer.add_scalar('test_mpjpe_pw3d', test_mpjpe_pw3d, epoch + 1)
                train_writer.add_scalar('test_pa_mpjpe_pw3d', test_pa_mpjpe_pw3d, epoch + 1)
                train_writer.add_scalar('test_mpve_pw3d', test_mpve_pw3d, epoch + 1)
            
            for k, v in losses_dict.items():
                train_writer.add_scalar('train_loss/'+k, v.avg, epoch + 1)
            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_mpjpe', mpjpes.avg, epoch + 1)
            train_writer.add_scalar('train_mpve', mpves.avg, epoch + 1)
                
            # Decay learning rate exponentially
            scheduler.step()
            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_jpe' : best_jpe
            }, chk_path)
            
            # Save checkpoint if necessary.
            if (epoch+1) % args.checkpoint_frequency == 0:
                chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
                print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_jpe' : best_jpe
            }, chk_path)

            if hasattr(args, "dt_file_pw3d"):
                best_jpe_cur = test_mpjpe_pw3d
            else:
                best_jpe_cur = test_mpjpe
            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if best_jpe_cur < best_jpe:
                best_jpe = best_jpe_cur
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_jpe' : best_jpe
                }, best_chk_path)

    if opts.evaluate:
        if hasattr(args, "dt_file_h36m"):
            test_loss, test_mpjpe, test_pa_mpjpe, test_mpve, _ = validate(test_loader, model, criterion, 'h36m')
        if hasattr(args, "dt_file_pw3d"):
            test_loss_pw3d, test_mpjpe_pw3d, test_pa_mpjpe_pw3d, test_mpve_pw3d, _ = validate(test_loader_pw3d, model, criterion, 'pw3d')

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)