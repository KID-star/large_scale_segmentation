import warnings
from helper_tool_car import ConfigCar as cfg
from Network import Proposed_network,compute_acc, IoUCalculator, compute_loss
from s3dis_mydataset_car import My_data, My_data_Sampler
import numpy as np
import os, argparse
import torch, gc
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time

torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='train_output', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 100]')
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 2], -1 for cpu')
parser.add_argument('--test_area', type=int, default=1, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()
LOG_DIR = FLAGS.log_dir
LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
log_file_name = f'log_train_Area_{FLAGS.test_area:d}.txt'
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

dataset = My_data(FLAGS.test_area)
training_dataset = My_data_Sampler(dataset, 'training')
validation_dataset = My_data_Sampler(dataset, 'validation')
training_dataloader= DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=training_dataset.collate_fn)
validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=validation_dataset.collate_fn)
print(len(training_dataloader), len(validation_dataloader))

if FLAGS.gpu >= 0:
    if torch.cuda.is_available():
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        FLAGS.gpu = torch.device('cpu')
else:
    FLAGS.gpu = torch.device('cpu')

device = FLAGS.gpu
net = Proposed_network(cfg)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
it = -1
start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))
def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']
    lr = lr * cfg.lr_decays[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def train_one_epoch():
    stat_dict = {}
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()
    iou_calc = IoUCalculator(cfg)
    for batch_idx, batch_data in enumerate(training_dataloader):
        t_start = time.time()
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        optimizer.zero_grad()
        end_points = net(batch_data)
        loss, end_points = compute_loss(end_points, cfg, device)
        loss.backward()
        optimizer.step()
        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
        batch_interval = 50
        if (batch_idx + 1) % batch_interval == 0:
            t_end = time.time()
            log_string('Step %03d Loss %.3f Acc %.2f lr %.5f --- %.2f ms/batch' % (batch_idx + 1, stat_dict['loss'] / batch_interval, stat_dict['acc'] / batch_interval, optimizer.param_groups[0]['lr'], 1000 * (t_end - t_start)))
            stat_dict['loss'], stat_dict['acc'] = 0, 0
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100))
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)


def evaluate_one_epoch():
    stat_dict = {}
    net.eval()
    iou_calc = IoUCalculator(cfg)
    for batch_idx, batch_data in enumerate(validation_dataloader):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        with torch.no_grad():
            end_points = net(batch_data)
        loss, end_points = compute_loss(end_points, cfg, device)
        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}%'.format(mean_iou * 100))
    log_string('--------------------------------------------------------------------------------------')
    s = f'{mean_iou*100:.1f} | '
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)
    log_string('--------------------------------------------------------------------------------------')
    return mean_iou


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    max_miou = 0
    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string(str(datetime.now()))
        np.random.seed()
        train_one_epoch()
        log_string('**** EVAL EPOCH %03d START****' % (epoch))
        now_miou = evaluate_one_epoch()
        if(now_miou>max_miou):
            save_dict = {'epoch': epoch+1,'optimizer_state_dict': optimizer.state_dict(),'loss': loss,}
            try:
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
            max_miou = now_miou
        log_string('Best mIoU = {:2.2f}%'.format(max_miou*100))
        log_string('**** EVAL EPOCH %03d END****' % (epoch))
        log_string('')

if __name__ == '__main__':

    train(start_epoch)

