import torch
import torch.nn.functional as F
import numpy as np
import argparse
from utils.dataloader import test_dataset
from utils.metrics import IOUMetric, DiceMetric, Re_Pre, MAEMetric
from lvss_network import SDNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=640, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./checkpoints/SDNet/bestf1.pth')

    opt = parser.parse_args()

    model = SDNet().cuda()

    weights = torch.load(opt.pth_path)
    model.load_state_dict(weights, strict=True)

    model.eval()

    image_root = "../defect/images/validation/"
    gt_root = "../defect/annotations/validation/"

    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    print(test_loader.size)

    iou_metric = IOUMetric()
    dice_metric = DiceMetric()
    re_pre_metric = Re_Pre(1)
    mae_metric = MAEMetric()
    iou_metric.reset()
    dice_metric.reset()
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

        mae_metric.update(pre_map, gt)
        iou_metric.update(pre_map, gt)
        dice_metric.update(pre_map, gt)
        pre_map = np.where(pre_map > 0.5, 1, 0)
        re_pre_metric.update(pre_map, gt)

    mae = mae_metric.get()
    pixAcc, IoU = iou_metric.get()
    dice = dice_metric.get()
    recall, precision, f1 = re_pre_metric.get()

    print("PixAcc / Iou / Dice / Recall / Precision / F1-Score / MAE : {} / {} / {} / {} / {} / {} / {}."
          .format(pixAcc, IoU, dice, recall, precision, f1, mae))



