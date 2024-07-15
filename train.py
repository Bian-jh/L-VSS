import torch
import torch.nn.functional as F
import os
import argparse
from datetime import datetime
from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter, set_seed
import numpy as np
import math
from utils.metrics import IOUMetric, Re_Pre
from lvss_network import SDNet


def structure_loss(pred, mask):
    weit = 1 + 10 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 为像素变化明显的地方加上更大的权重
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model):
    model.eval()
    iou_metric = IOUMetric()
    iou_metric.reset()
    re_pre_metric = Re_Pre(1)

    image_root = "../defect/images/validation/"
    gt_root = "../defect/annotations/validation/"
    test_loader = test_dataset(image_root, gt_root, 640)
    print('[test_size]', test_loader.size)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.array(gt)
        gt = np.where(gt > 127, 1, 0)
        with torch.no_grad():
            image = image.cuda()

            pre_map, pre_map_1, pre_map_2, pre_map_3, pre_map_4 = model(image)
            pre_map = F.interpolate(pre_map, size=gt.shape, mode='bilinear', align_corners=False)
            pre_map = pre_map.sigmoid().data.cpu().numpy().squeeze()
            pre_map = (pre_map - pre_map.min()) / (pre_map.max() - pre_map.min() + 1e-8)

        iou_metric.update(pre_map, gt)
        pre_map = np.where(pre_map > 0.5, 1, 0)
        re_pre_metric.update(pre_map, gt)

    pixAcc, IoU = iou_metric.get()
    recall, precision, f1 = re_pre_metric.get()

    return IoU, recall, precision, f1


def train(opt, train_loader, model, optimizer, epoch, total_step, best_f1, best_iou):
    model.train()
    loss_record, loss_record1, loss_record2,  loss_record3, loss_record4 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()

        # ---- forward ----
        pre_map, pre_map_1, pre_map_2, pre_map_3, pre_map_4 = model(images)
        # ---- loss function ----
        loss1 = structure_loss(pre_map, gts)
        loss2 = structure_loss(pre_map_1, gts)
        loss3 = structure_loss(pre_map_2, gts)
        loss4 = structure_loss(pre_map_3, gts)
        loss5 = structure_loss(pre_map_4, gts)

        loss = loss1 + loss2 + loss3 + loss4 + loss5
        # ---- backward ----
        loss.backward()
        optimizer.step()
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

        # ---- recording loss ----
        loss_record.update(loss1.data, opt.batchsize)
        loss_record1.update(loss2.data, opt.batchsize)
        loss_record2.update(loss3.data, opt.batchsize)
        loss_record3.update(loss4.data, opt.batchsize)
        loss_record4.update(loss5.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  'predict: {:0.4f}], predict-1: {:0.4f}], predict-2: {:0.4f}], predict-3: {:0.4f}], predict-4: {:0.4f}], now-lr: {:0.6f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(), loss_record1.show(), loss_record2.show(), loss_record3.show(), loss_record4.show(), now_lr))
    save_path = 'checkpoints/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)

    # test model performance
    IoU, recall, precision, f1 = test(model)
    print('[IoU:]', IoU, '[Recall:]', recall, '[Precision:]', precision, '[f1:]', f1)

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), save_path + 'bestf1.pth')
        print('[Saving bestf1 checkpoint:]', save_path + 'bestf1.pth', '[best f1:]', best_f1, '[Recall:]', recall, '[Precision:]', precision)

    if IoU > best_iou:
        best_iou = IoU
        torch.save(model.state_dict(), save_path + 'bestIoU.pth')
        print('[Saving bestIoU checkpoint:]', save_path + 'bestIoU.pth', '[best IoU:]', best_iou)

    return best_f1, best_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=640, help='training dataset size')

    parser.add_argument('--train_save', type=str,
                        default='SDNet')

    opt = parser.parse_args()

    set_seed(0)
    # ---- build models ----
    # model = torch.nn.DataParallel(SDNet(), device_ids=[0]).cuda()
    model = SDNet().cuda()

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    params = model.parameters()

    optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)

    print(optimizer)

    warm_up_epochs = 10
    lr_func = lambda epoch: (epoch+1) / warm_up_epochs if (epoch+1) <= warm_up_epochs else 0.5 * (
    math.cos((epoch + 1 - warm_up_epochs) / (opt.epoch - warm_up_epochs) * math.pi) + 1)
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

    image_root = "../defect/images/training/"
    gt_root = "../defect/annotations/training/"

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    best_f1 = 0
    best_iou = 0
    for epoch in range(1, opt.epoch + 1):
        torch.cuda.empty_cache()
        best_f1, best_iou = train(opt, train_loader, model, optimizer, epoch, total_step, best_f1, best_iou)
        lr_schedule.step()

