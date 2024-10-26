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
from lib.data.dataset_alphapose import AlphaPoseDataset
from lib.model.model_walking_rnn import ActionNet

from torch.nn.utils.rnn import pack_padded_sequence

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/walking/MB_ft_walking.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint/walking', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint/pretrain/MB_release', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-freq', '--print_freq', default=1)
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('--kcv', default=False, type=bool, metavar='BOOL', help='k-fold cross validation')
    opts = parser.parse_args()
    return opts

def validate(test_loader, model, criterion):
    model.eval()
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        end = time.time()
        for idx, (batch_input, batch_gt) in tqdm(enumerate(test_loader)):
            batch_size = len(batch_input)
            if torch.cuda.is_available():
                batch_gt = batch_gt.cuda()
                batch_input = batch_input.cuda()
            output = model(batch_input)    # (N, num_classes)
            loss = criterion(output, batch_gt)
            acc = binary_accuracy(output, batch_gt)

            # update metric
            losses.update(loss.item(), batch_size)
            accs.update(acc, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (idx+1) % opts.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                       idx, len(test_loader), batch_time=batch_time,
                       loss=losses, acc=accs))
    return losses.avg, accs.avg

def train_with_config(args, opts):
    print('INFO: Training with all data')
    all_json_paths, labels = get_data(os.path.join('data', 'walking'))
    train_json_paths, train_labels, test_json_paths, test_labels = all_json_paths, labels, all_json_paths, labels
    print(args)
    try:
        os.makedirs(os.path.join(opts.checkpoint, "all"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "all", "logs"))
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
    model = WalkingNet(backbone=model_backbone, dim_rep=args.dim_rep, num_classes=args.action_classes, dropout_ratio=args.dropout_ratio, version=args.model_version, hidden_dim=args.hidden_dim, num_joints=args.num_joints)
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
    best_acc = 0
    model_params = 0
    for parameter in model.parameters():
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
    # data_path = 'data/action/%s.pkl' % args.dataset

    train_alphapose_dataset = AlphaPoseDataset(train_json_paths, train_labels, train=True, n_frames=243, random_move=True, scale_range=[1,1], check_split=True)
    test_alphapose_dataset = AlphaPoseDataset(test_json_paths, test_labels, train=False, n_frames=243, random_move=True, scale_range=[1,1], check_split=True)

    train_loader = DataLoader(train_alphapose_dataset, **trainloader_params)
    test_loader = DataLoader(test_alphapose_dataset, **testloader_params)

    chk_filename = os.path.join(opts.checkpoint, "all", "latest_epoch.bin")
    if os.path.exists(chk_filename):
        opts.resume = chk_filename
    if opts.resume or opts.evaluate:
        chk_filename = opts.evaluate if opts.evaluate else opts.resume
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        # dont load the model
        # model.load_state_dict(checkpoint['model'], strict=True)

    if not opts.evaluate:
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
        all_accs_train, all_loss_train, all_accs_test, all_loss_test = [], [], [], []
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            losses_train = AverageMeter()
            accs_train = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            model.train()
            end = time.time()
            iters = len(train_loader)
            for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):    # (N, 2, T, 17, 3)
                data_time.update(time.time() - end)
                batch_size = len(batch_input)
                if torch.cuda.is_available():
                    batch_gt = batch_gt.cuda()
                    batch_input = batch_input.cuda()
                output = model(batch_input) # (N, num_classes)
                optimizer.zero_grad()
                loss_train = criterion(output, batch_gt)
                losses_train.update(loss_train.item(), batch_size)
                acc_train = binary_accuracy(output, batch_gt)
                accs_train.update(acc_train, batch_size)
                loss_train.backward()
                optimizer.step()
                batch_time.update(time.time() - end)
                end = time.time()
            if (idx + 1) % opts.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                    'Binary_Acc {accs_train.val:.3f} ({accs_train.avg:.3f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses_train, accs_train=accs_train))
                sys.stdout.flush()

            test_loss, test_acc = validate(test_loader, model, criterion)

            train_writer.add_scalar('train_loss', losses_train.avg, epoch + 1)
            train_writer.add_scalar('train_acc', accs_train.avg, epoch + 1)
            train_writer.add_scalar('test_loss', test_loss, epoch + 1)
            train_writer.add_scalar('test_acc', test_acc, epoch + 1)
            all_accs_train.append(accs_train.avg)
            all_loss_train.append(losses_train.avg)
            all_accs_test.append(test_acc)
            all_loss_test.append(test_loss)

            scheduler.step()

            # Save latest checkpoint.
            chk_path = os.path.join(opts.checkpoint, "all", 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
            }, chk_path)

            # Save best checkpoint.
            best_chk_path = os.path.join(opts.checkpoint, "all", 'best_epoch.bin'.format(epoch))
            if test_acc > best_acc:
                best_acc = test_acc
                print("save best checkpoint")
                torch.save({
                'epoch': epoch+1,
                'lr': scheduler.get_last_lr(),
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'best_acc' : best_acc
                }, best_chk_path)

        # display as image
        display_train_test_results(os.path.join("vis"), "all", all_accs_train, all_loss_train, all_accs_test, all_loss_test)

    if opts.evaluate:
        test_loss, test_acc, test_top5 = validate(test_loader, model, criterion)
        print('Loss {loss:.4f} \t'
              'Acc@1 {top1:.3f} \t'.format(loss=test_loss, top1=test_acc))

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(captions, threshold):
    """Build a simple vocabulary wrapper."""
    counter = {}
    for i, caption in enumerate(captions):
        tokens = caption.split(' ')
        if len(tokens) == 0:
            print('WARNING: Found empty caption')
            continue
        for token in tokens:
            if token not in counter:
                counter[token] = 0
            counter[token] += 1

    words = [word for word in counter if counter[word] >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab

from torch.utils.data import Dataset
from lib.utils.utils_data import crop_scale, resample

def halpe2h36m(x):
    '''
        Input: x (T x V x C)
       //Halpe 26 body keypoints
    {0,  "Nose"},
    {1,  "LEye"},
    {2,  "REye"},
    {3,  "LEar"},
    {4,  "REar"},
    {5,  "LShoulder"},
    {6,  "RShoulder"},
    {7,  "LElbow"},
    {8,  "RElbow"},
    {9,  "LWrist"},
    {10, "RWrist"},
    {11, "LHip"},
    {12, "RHip"},
    {13, "LKnee"},
    {14, "Rknee"},
    {15, "LAnkle"},
    {16, "RAnkle"},
    {17,  "Head"},
    {18,  "Neck"},
    {19,  "Hip"},
    {20, "LBigToe"},
    {21, "RBigToe"},
    {22, "LSmallToe"},
    {23, "RSmallToe"},
    {24, "LHeel"},
    {25, "RHeel"},
    '''
    T, V, C = x.shape
    y = np.zeros([T,17,C])
    y[:,0,:] = x[:,19,:]
    y[:,1,:] = x[:,12,:]
    y[:,2,:] = x[:,14,:]
    y[:,3,:] = x[:,16,:]
    y[:,4,:] = x[:,11,:]
    y[:,5,:] = x[:,13,:]
    y[:,6,:] = x[:,15,:]
    y[:,7,:] = (x[:,18,:] + x[:,19,:]) * 0.5
    y[:,8,:] = x[:,18,:]
    y[:,9,:] = x[:,0,:]
    y[:,10,:] = x[:,17,:]
    y[:,11,:] = x[:,5,:]
    y[:,12,:] = x[:,7,:]
    y[:,13,:] = x[:,9,:]
    y[:,14,:] = x[:,6,:]
    y[:,15,:] = x[:,8,:]
    y[:,16,:] = x[:,10,:]
    return y

def read_input(json_path, vid_size, scale_range, focus):
    with open(json_path, "r") as read_file:
        results = json.load(read_file)
    kpts_all = []
    for item in results:
        if focus!=None and item['idx']!=focus:
            continue
        kpts = np.array(item['keypoints']).reshape([-1,3])
        kpts_all.append(kpts)
    kpts_all = np.array(kpts_all)
    kpts_all = halpe2h36m(kpts_all)
    if vid_size:
        w, h = vid_size
        scale = min(w,h) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] - np.array([w, h]) / 2.0
        kpts_all[:,:,:2] = kpts_all[:,:,:2] / scale
        motion = kpts_all
    if scale_range:
        motion = crop_scale(kpts_all, scale_range)
    return motion.astype(np.float32)

class AlphaPoseAnnotDataset(Dataset):
    def __init__(self, json_paths, captions, vocab, train=True, n_frames=243, random_move=True, scale_range=[1,1]):
        self.json_paths = json_paths
        self.captions = captions
        self.vocab = vocab
        self.train = train
        self.n_frames = n_frames
        self.random_move = random_move
        self.scale_range = scale_range
        self.X = []

        self._process_json()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.json_paths)

    def _process_json(self):
        """
        Process the json files and store the data in self.X
        """
        for json_path in self.json_paths:
            motion = np.array(read_input(json_path, vid_size=None, scale_range=self.scale_range, focus=None))
            resample_id = resample(ori_len=motion.shape[0], target_len=self.n_frames, randomness=True)
            motion = motion[resample_id]
            fake = np.zeros(motion.shape)
            motion = np.array([motion, fake])
            # change to tensor
            self.X.append(torch.tensor(motion.astype(np.float32)))

    def __getitem__(self, index):
        """
        Returns a sample of data
        self.X[index]: (2, n_frames, 17, 3)
        self.y[index]: label (0 or 1)
        """
        caption = self.captions[index]
        caption = caption.split(' ')
        caption = ['<start>'] + caption + ['<end>']
        caption = [self.vocab(word) for word in caption]
        return self.X[index], torch.tensor(caption)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seg_length = max_seq_length

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    motion, captions = zip(*data)
    motion = torch.stack(motion, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return motion, targets, lengths


def train_rnn(args, opts):
    json_paths = [
        'data/walking/model/json/1.json',
        'data/walking/model/json/2.json',
        'data/walking/model/json/3.json',
        'data/walking/model/json/6.json',
        'data/walking/model/json/5.json',
    ]*10
    captions = [
        "Model walks confidently with a fierce gaze and hands on hips.",
        "Model makes eye contact with the audience and gives a slight nod.",
        "Model struts with shoulders back, radiating poise and elegance.",
        "Model turns to the side and glances over their shoulder with a smirk.",
        "Model gives a playful wink to the camera while walking forward.",
        "Model holds the edge of their outfit, showing off intricate details.",
        "Model pauses mid-walk, placing one hand on the hip and tilting the head.",
        "Model twirls gracefully, showcasing the outfit's movement.",
        "Model looks up with a serene expression, embodying calmness.",
        "Model waves gently at the audience before resuming the walk.",
        "Model holds a prop and presents it while smiling at the crowd.",
        "Model gives a quick spin, allowing the fabric to flow.",
        "Model strides with purpose, making direct eye contact with the camera.",
        "Model slows down, placing one foot slightly forward for a dramatic pause.",
        "Model places both hands on the waist and poses assertively.",
        "Model raises one hand to blow a kiss toward the camera.",
        "Model does a half turn, flashing a smile back at the audience.",
        "Model stops to adjust an accessory, maintaining an elegant posture.",
        "Model points one foot forward, creating a sense of motion.",
        "Model moves hands behind the back, adding a sophisticated touch.",
        "Model stops to sway from side to side, giving a full view of the outfit.",
        "Model raises eyebrows and gives a confident, playful smile.",
        "Model pauses, gazing off into the distance with a soft expression.",
        "Model gives a peace sign toward the audience before continuing the walk.",
        "Model gently touches their necklace, drawing attention to the jewelry.",
        "Model tilts the head slightly and runs fingers through their hair.",
        "Model turns both palms outward, expressing openness.",
        "Model shifts weight onto one leg, leaning slightly to add attitude.",
        "Model tilts chin up, exuding confidence and strength.",
        "Model brings one hand up to cover their mouth in a coy gesture.",
        "Model winks playfully at a fellow model on the runway.",
        "Model walks with arms crossed, adding a touch of defiance.",
        "Model holds skirt edges delicately, lifting them slightly while walking.",
        "Model smiles warmly while glancing side to side at the audience.",
        "Model spins while moving forward, creating a flowing motion.",
        "Model bows slightly, acknowledging the crowd.",
        "Model waves excitedly, bringing energy to the walk.",
        "Model walks with a hand casually placed in a pocket.",
        "Model gestures to an accessory, drawing attention to its details.",
        "Model walks with exaggerated steps, emphasizing the outfit's fit.",
        "Model stops at the runway's end, placing one foot forward assertively.",
        "Model lifts an arm in a victorious gesture, radiating pride.",
        "Model clasps hands together and gives a big, genuine smile.",
        "Model brings one finger to lips, signaling playful mystery.",
        "Model moves with a slow, deliberate strut, giving a relaxed vibe.",
        "Model gives a quick two-finger salute to the audience.",
        "Model lifts one shoulder while looking back with a sly grin.",
        "Model mimics a snap, adding a touch of flair.",
        "Model strikes a quick pose with arms stretched out, showing the outfit's cut.",
        "Model places a hand on the chin and tilts head, lost in thought."
    ]
    vocab = build_vocab(captions, threshold=1)
    vocab_size = len(vocab)
    dataset = AlphaPoseAnnotDataset(json_paths, captions, vocab, train=True, n_frames=243, random_move=True, scale_range=[1,1])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    model_backbone = load_backbone(args)
    model = ActionNet(backbone=model_backbone, dim_rep=args.dim_rep, dropout_ratio=args.dropout_ratio, hidden_dim=args.hidden_dim, num_joints=args.num_joints)

    decoder = DecoderRNN(embed_size=2048, hidden_size=512, vocab_size=vocab_size, num_layers=1)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()
        decoder = decoder.cuda()
        criterion = criterion.cuda()

    num_epochs = 10
    for epoch in range(num_epochs):
        for i, (motion, captions, lengths) in enumerate(dataloader):
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            if torch.cuda.is_available():
                motion = motion.cuda()
                captions = captions.cuda()
                targets = targets.cuda()
            features = model(motion)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, i, len(dataloader), loss.item()))
                print()

    print('Finished training')
    print("Testing with a sample")
    test_json = 'data/walking/model/json/7.json'
    test_data = read_input(test_json, vid_size=None, scale_range=[1,1], focus=None)
    test_data = test_data.astype(np.float32)
    resample_id = resample(ori_len=test_data.shape[0], target_len=243, randomness=False)
    test_data = test_data[resample_id]
    fake = np.zeros(test_data.shape)
    test_data = np.array([[test_data, fake]]).astype(np.float32)
    sampled_ids = decoder.sample(model(torch.tensor(test_data).cuda()))
    sampled_ids = sampled_ids[0].cpu().numpy()
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    print(sentence)

if __name__ == "__main__":
    opts = parse_args()
    args = get_config(opts.config)
    # train_with_config(args, opts)
    train_rnn(args, opts)
