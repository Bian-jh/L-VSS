import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
from utils.dataloader import test_dataset
from lvss_network import SDNet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=640, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/SDNet/bestf1.pth')


save_path = "./My_results/"
opt = parser.parse_args()

model = SDNet()
weights = torch.load(opt.pth_path)
model.load_state_dict(weights)
model.cuda()
model.eval()

os.makedirs(save_path, exist_ok=True)
image_root = "../defect/images/validation/"
gt_root = "../defect/annotations/validation/"
test_loader = test_dataset(image_root, gt_root, opt.testsize)

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()

    res, _, _, _, _ = model(image)
    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res *= 255
    res = np.where(res > 127, 255, 0)
    res = res.astype(np.uint8)

    imageio.imwrite(save_path + name, res)