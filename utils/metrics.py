import numpy as np
from skimage import measure
from sklearn.metrics import auc


# (x_left, y_top, x_right, y_bottom)
def get_IoU(bbox1, bbox2):
    # 步骤一：获取交集部分坐标
    ix_min = max(bbox1[0], bbox2[0])
    iy_min = max(bbox1[1], bbox2[1])
    ix_max = min(bbox1[2], bbox2[2])
    iy_max = min(bbox1[3], bbox2[3])

    iw = max(ix_max - ix_min, 0.0)
    ih = max(iy_max - iy_min, 0.0)

    inters = iw * ih

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    uni = area1 + area2 - inters

    overlaps = inters / uni
    return overlaps


class IOUMetric():
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape

        predict = (output > 0.5).astype('int64') # P
        pixel_labeled = np.sum(target > 0) # T
        pixel_correct = np.sum((predict == target)*(target > 0)) # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1 # nclass
        nbins = 1 # nclass
        predict = (output > 0.5).astype('int64') # P
        target = target.astype('int64') # T
        intersection = predict * (predict == target) # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union


class DiceMetric():
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        inter, sum = self.batch_intersection(pred, labels)

        self.total_inter += inter
        self.total += sum

    def get(self):
        """Gets the current evaluation result."""
        smooth = 1
        dice = (2 * self.total_inter + smooth) / (self.total + smooth)
        return dice

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total = 0

    def batch_intersection(self, output, target):
        output = (output > 0.5).astype('int64')
        input_flat = np.reshape(output, (-1))
        target_flat = np.reshape(target, (-1))

        intersection = (input_flat * target_flat)

        return intersection.sum(), output.sum() + target.sum()


class Re_Pre():
    def __init__(self, nclass):
        self.nclass = nclass
        self.TP = 0
        self.target = 0
        self.P = 0

    def update(self, preds, labels):
        image = measure.label(preds, connectivity=2)
        coord_image = measure.regionprops(image)
        label = measure.label(labels, connectivity=2)
        coord_label = measure.regionprops(label)

        image_area_total = []
        for K in range(len(coord_label)):
            image_area_total.append(coord_label[K].area)
        ind = image_area_total.index(max(image_area_total))  # filter out irrelevant regions
        self.target += 1
        self.P += len(coord_image)

        y_min, x_min, y_max, x_max = coord_label[ind].bbox

        for m in range(len(coord_image)):
            y1, x1, y2, x2 = coord_image[m].bbox
            overlaps = get_IoU([x_min, y_min, x_max, y_max], [x1, y1, x2, y2])
            if overlaps > 0.5:
                self.TP += 1

                break

    def get(self):
        print(self.target)
        recall = self.TP / (self.target + 1e-10)
        precision = self.TP / (self.P + 1e-10)
        f1 = (2 * precision * recall) / (precision + recall + 1e-10)

        return recall, precision, f1


class PRMetric():
    def __init__(self, bins=10):
        self.bins = bins
        self.TP = np.zeros(self.bins + 1)
        self.target = np.zeros(self.bins + 1)
        self.P = np.zeros(self.bins + 1)

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            predict = (preds > score_thresh).astype('int64')

            image = measure.label(predict, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labels, connectivity=2)
            coord_label = measure.regionprops(label)

            image_area_total = []
            for K in range(len(coord_label)):
                image_area_total.append(coord_label[K].area)
            ind = image_area_total.index(max(image_area_total))  # filter out irrelevant regions
            self.target[iBin] += 1
            self.P[iBin] += len(coord_image)

            y_min, x_min, y_max, x_max = coord_label[ind].bbox

            for m in range(len(coord_image)):
                y1, x1, y2, x2 = coord_image[m].bbox
                overlaps = get_IoU([x_min, y_min, x_max, y_max], [x1, y1, x2, y2])
                if overlaps > 0.5:
                    self.TP[iBin] += 1

                    break

    def get(self):
        recall = self.TP / (self.target + 1e-10)
        precision = self.TP / (self.P + 1e-10)

        return recall, precision, auc(recall, precision)


class MAEMetric():
    def __init__(self):
        self.error = 0

    def update(self, preds, labels):
        # predict = (preds > 0.5).astype('int64')
        predict = preds

        target = labels.astype('int64')
        self.error += np.sum(np.abs(predict - target)) / (target.shape[0] * target.shape[1])

    def get(self):
        mae = self.error

        return mae
