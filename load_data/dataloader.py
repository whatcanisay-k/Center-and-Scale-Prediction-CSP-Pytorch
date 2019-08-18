import sys
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from load_data.load_data import get_citypersons


class CityPersons(Dataset):
    def __init__(self, path, type, config):

        self.dataset = get_citypersons(root_dir=path, type=type)
        self.dataset_len = len(self.dataset)
        self.type = type
        self.radius = config.radius
        self.stride = config.stride
        self.size = config.train_size


    def __getitem__(self, item):

        # input is RGB order, and normalized
        img_data = self.dataset[item]

        img = np.float32(cv2.imread(img_data['filepath'], 1))
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)

        if self.type == 'train':
            boxes = img_data['bboxes'].copy()

            img, boxes = self.random_crop(img, boxes, self.size, limit=16)
            mask, scale_map, offset_map = self.get_label(boxes, img.shape)

            return img, mask, scale_map, offset_map

        else:
            return img

    def __len__(self):
        return self.dataset_len

    def gaussian(self, kernel):
        sigma = ((kernel - 1) * 0.3 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    def get_label(self, boxes, img_shape):

        img_shape = [img_shape[1], img_shape[2], img_shape[0]]
        mask = np.zeros((3, int(img_shape[0]/self.stride), int(img_shape[1]/self.stride)))
        scale_map = np.zeros((3, int(img_shape[0]/self.stride), int(img_shape[1]/self.stride)))
        offset_map = np.zeros((3, int(img_shape[0]/self.stride), int(img_shape[1]/self.stride)))
        mask[1, :, :] = 1

        if len(boxes)>0:
            boxes = boxes / self.stride
            for box in boxes:
                x1, y1, x2, y2 = [int(x) for x in box]

                c_x, c_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                dx = self.gaussian(x2 - x1)
                dy = self.gaussian(y2 - y1)
                gau_map = np.multiply(dy, np.transpose(dx))

                mask[0, y1:y2, x1:x2] = np.maximum(mask[0, y1:y2, x1:x2], gau_map)
                mask[1, y1:y2, x1:x2] = 1
                mask[2, c_y, c_x] = 1

                scale_map[0, c_y - self.radius:c_y + self.radius + 1, c_x - self.radius:c_x + self.radius + 1] = np.log(
                    (y2 - y1))  # log value of height
                scale_map[1, c_y - self.radius:c_y + self.radius + 1, c_x - self.radius:c_x + self.radius + 1] = np.log(
                    (x2 - x1))  # log value of width

                scale_map[2, c_y - self.radius:c_y + self.radius + 1, c_x - self.radius:c_x + self.radius + 1] = 1

                offset_map[0, c_y, c_x] = (y1 + y2) / 2 - c_y - 0.5
                offset_map[1, c_y, c_x] = (x1 + x2) / 2 - c_x - 0.5
                offset_map[2, c_y, c_x] = 1

        return mask, scale_map, offset_map

    def random_crop(self, img, boxes, size, limit=8):
        _, h, w = img.shape
        crop_h, crop_w = size

        '''if len(boxes) > 0:
            sel_id = np.random.randint(0, len(boxes))
            sel_center_x = int((boxes[sel_id, 0] + boxes[sel_id, 2]) / 2.0)
            sel_center_y = int((boxes[sel_id, 1] + boxes[sel_id, 3]) / 2.0)
        else:'''
        sel_center_x = int(np.random.randint(0, w - crop_w + 1) + crop_w * 0.5)
        sel_center_y = int(np.random.randint(0, h - crop_h + 1) + crop_h * 0.5)

        crop_x1 = max(sel_center_x - int(crop_w * 0.5), int(0))
        crop_y1 = max(sel_center_y - int(crop_h * 0.5), int(0))
        diff_x = max(crop_x1 + crop_w - w, int(0))
        crop_x1 -= diff_x
        diff_y = max(crop_y1 + crop_h - h, int(0))
        crop_y1 -= diff_y
        cropped_img = img[:, crop_y1:(crop_y1 + crop_h), crop_x1:(crop_x1 + crop_w)]


        if len(boxes) > 0:
            before_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes[:, 0:4:2] -= crop_x1
            boxes[:, 1:4:2] -= crop_y1
            boxes[:, 0:4:2] = np.clip(boxes[:, 0:4:2], 0, crop_w)
            boxes[:, 1:4:2] = np.clip(boxes[:, 1:4:2], 0, crop_h)

            after_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            keep_inds = ((boxes[:, 2] - boxes[:, 0]) >= limit) & (after_area >= 0.5 * before_area)
            boxes = boxes[keep_inds]

        return cropped_img, boxes
