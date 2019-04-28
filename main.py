# requirements differences:
# - python 3.7
# - pytorch 1
# - logger: Wandb

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger

# ==== new imports === #
import numpy as np
import wandb

# ==== fix seeds === #
torch.manual_seed(0)
np.random.seed(0)


import models.anynet

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=4,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')


args = parser.parse_args()

# =========================== set device ============================== #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# ============ LOGGER ============ #
wandb.init(project="CLONED_ANYNET_MAIN")
wandb.config.update(args)


def main():
    global args

    # ==== INFO ==== #
    print(torch.__version__)
    print('running on: ', device)
    print('running gpus: ', torch.cuda.device_count())

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = logger.setup_logger(args.save_path + '/training.log')
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).to(device)  #.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')

    start_full_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

    test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().to(device)      # .cuda()
        imgR = imgR.float().to(device)      # .cuda()
        disp_L = disp_L.float().to(device)  # .cuda()

        optimizer.zero_grad()
        mask = disp_L < args.maxdisp
        mask.detach_()
        outputs = model(imgL, imgR)
        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(stages):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])

        # ============ new logs ============================================= #
        if not batch_idx % 450:
            wandb.log({"train left RGB":      (wandb.Image(imgL[0])),
                       "train disp stage 1":  (wandb.Image(outputs[0][0])),
                       "train disp stage 2":  (wandb.Image(outputs[1][0])),
                       "train disp stage 3":  (wandb.Image(outputs[2][0])),
                       "train disp gt":       (wandb.Image(disp_L[0])),
                        })

        if not batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)

            # ============ new logs ======================== #
            wandb.log({"train loss stage 1": losses[0].val,
                       "train loss stage 2": losses[1].val,
                       "train loss stage 3": losses[2].val})

            log.info('Epoch{} [{}/{}] {}'.format(
                epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

    # ============ new logs ================================= #
    wandb.log({"final avg train loss stage 1": losses[0].avg,
               "final avg train loss stage 2": losses[1].avg,
               "final avg train loss stage 3": losses[2].avg})


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    inference_time = []
    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().to(device)  # .cuda()
        imgR = imgR.float().to(device)  # .cuda()
        disp_L = disp_L.float().to(device)  # .cuda()

        mask = disp_L < args.maxdisp
        with torch.no_grad():

            # === take time === #
            time_before = time.time()
            outputs = model(imgL, imgR)
            inference_time.append(time.time() - time_before)

            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
                output = output[:, 4:, :]
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())

        # ================= new logs ===================================== #
        if not batch_idx % 100:
            wandb.log({"test left RGB":  (wandb.Image(imgL[0])),
                       "test disp stage 1":   (wandb.Image(outputs[0][0])),
                       "test disp stage 2":   (wandb.Image(outputs[1][0])),
                       "test disp stage 3":   (wandb.Image(outputs[2][0])),
                       "test disp gt":  (wandb.Image(disp_L[0]))})

        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])

        log.info('[{}/{}] {}'.format(
            batch_idx, length_loader, info_str))

        # ========== new logs ======================== #
        wandb.log({"EPEs stage 1": EPEs[0].val,
                   "EPEs stage 2": EPEs[1].val,
                   "EPEs stage 3": EPEs[2].val,
                   "avg EPEs stage 0": EPEs[0].avg,
                   "avg EPEs stage 1": EPEs[1].avg,
                   "avg EPEs stage 2": EPEs[2].avg})

    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    log.info('Average test EPE = ' + info_str)

    # ========== new logs ============================ #
    wandb.log({"final avg EPEs stage 0": EPEs[0].avg,
               "final avg EPEs stage 1": EPEs[1].avg,
               "final avg EPEs stage 2": EPEs[2].avg,
               "avg inference time": np.mean(inference_time)})


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
