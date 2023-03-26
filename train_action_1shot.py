import os
import numpy as np
import time
import sys
import argparse
import errno
from collections import OrderedDict
import tensorboardX
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.model.loss import *
from lib.data.dataset_action import NTURGBD, NTURGBD1Shot
from lib.model.model_action import ActionNet

from lib.model.loss_supcon import SupConLoss
from pytorch_metric_learning import samplers

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=100)
    parser.add_argument('-ms', '--selection', default='best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    opts = parser.parse_args()
    return opts

def extract_feats(dataloader_x, model):
    all_feats = []
    all_gts = []
    with torch.no_grad():
        for idx, (batch_input, batch_gt) in tqdm(enumerate(dataloader_x)):    # (N, 2, T, 17, 3)
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()      
            feat = model(batch_input)
            all_feats.append(feat)
            all_gts.append(batch_gt)
    all_feats = torch.cat(all_feats)
    all_gts = torch.cat(all_gts)
    return all_feats, all_gts

def validate(anchor_loader, test_loader, model):
    train_feats, train_labels = extract_feats(anchor_loader, model)
    test_feats, test_labels = extract_feats(test_loader, model)
    M = len(train_feats)
    N = len(test_feats)
    train_feats = train_feats.unsqueeze(1)
    test_feats = test_feats.unsqueeze(0)
    dis = F.cosine_similarity(train_feats, test_feats, dim=-1)
    pred = train_labels[torch.argmax(dis, dim=0)]
    assert len(pred)==len(test_labels)
    acc = sum(pred==test_labels) / len(pred)
    return acc

def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))
    model_backbone = load_backbone(args)
    if args.finetune:
        if opts.resume or opts.evaluate:
            pass
        else:
            chk_filename = os.path.join(opts.pretrained, "best_epoch.bin")
            print('Loading backbone', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model_pos'].items():
                name = k[7:]                                            # remove 'module.'
                new_state_dict[name] = v 
            model_backbone.load_state_dict(new_state_dict, strict=True)
            if args.partial_train:
                model_backbone = partial_train_layers(model_backbone, args.partial_train)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = SupConLoss(temperature=args.temp)
    
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        
    chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'], strict=True)
        
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)
    print('Loading dataset...')

    anchorloader_params = {
              'batch_size': args.batch_size,
              'shuffle': False,
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
    data_path_1shot = 'data/action/ntu120_hrnet_oneshot.pkl'
    ntu60_1shot_anchor = NTURGBD(data_path=data_path_1shot, data_split='oneshot_train', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)
    ntu60_1shot_test = NTURGBD(data_path=data_path_1shot, data_split='oneshot_val', n_frames=args.clip_len, random_move=False, scale_range=args.scale_range_test)
    anchor_loader = DataLoader(ntu60_1shot_anchor, **anchorloader_params)
    test_loader = DataLoader(ntu60_1shot_test, **testloader_params)

    if not opts.evaluate:    
        # Load training data (auxiliary set)
        data_path = 'data/action/ntu120_hrnet.pkl'
        ntu120_1shot_train = NTURGBD1Shot(data_path=data_path, data_split='', n_frames=args.clip_len, random_move=args.random_move, scale_range=args.scale_range_train, check_split=False)
        sampler = samplers.MPerClassSampler(ntu120_1shot_train.labels, m=args.n_views, batch_size=args.batch_size, length_before_new_iter=len(ntu120_1shot_train))
        trainloader_params = {
              'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': 8,
              'pin_memory': True,
              'prefetch_factor': 4,
              'persistent_workers': True,
              'sampler': sampler
        }
        train_loader = DataLoader(ntu120_1shot_train, **trainloader_params)
        optimizer = optim.AdamW(
            [     {"params": filter(lambda p: p.requires_grad, model.module.backbone.parameters()), "lr": args.lr_backbone},
                  {"params": filter(lambda p: p.requires_grad, model.module.head.parameters()), "lr": args.lr_head},
            ],      lr=args.lr_backbone, 
                    weight_decay=args.weight_decay
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=args.lr_decay)
        st = 0
        print('INFO: Training on {} batches'.format(len(train_loader)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

            lr = checkpoint['lr']
            if 'best_acc' in checkpoint and checkpoint['best_acc'] is not None:
                best_acc = checkpoint['best_acc']
                
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            model.train()
            end = time.time()
            
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                feat = model(batch_input) 
                feat = feat.reshape(batch_size, -1, args.hidden_dim)
                optimizer.zero_grad()
                loss_train = criterion(feat, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                loss_train.backward()
                optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'.format(
                       epoch, idx + 1, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses_train))
                sys.stdout.flush()
            test_top1 = validate(anchor_loader, test_loader, model)
            train_writer.add_scalar('train_loss_supcon', losses_train.avg, epoch + 1)
            train_writer.add_scalar('test_top1', test_top1, epoch + 1)
            scheduler.step()
            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint
            best_chk_path = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            if test_top1 > best_acc:
                best_acc = test_top1
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)
    if opts.evaluate:
        test_top1 = validate(anchor_loader, test_loader, model)
        print(test_top1)
if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    train_with_config(args, opts)
    