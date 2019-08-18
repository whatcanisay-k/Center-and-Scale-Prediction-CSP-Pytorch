import numpy as np
import math
import cv2

img_height = 812
img_width= 1090
pi = math.pi

class DataLoader(object):
    def __init__(self, radius):
        self.radius = radius


    def gaussian(self, kernel):
        sigma = ((kernel - 1) * 0.2 - 1) * 0.3 + 0.8
        s = 2 * (sigma ** 2)
        dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
        return np.reshape(dx, (-1, 1))

    def get_label(self, boxes, img_shape):

        img_shape = [int(i/4) for i in img_shape]
        mask = np.zeros((3, img_shape[0], img_shape[1]))
        scale_map = np.zeros((3, img_shape[0], img_shape[1]))
        offset_map = np.zeros((3, img_shape[0], img_shape[1]))

        for box in boxes:
            x1, y1, x2, y2 = [int(x/4) for x in box]
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img_shape[1]-1), min(y2, img_shape[0]-1)
            c_x, c_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            dx = self.gaussian(x2 - x1)
            dy = self.gaussian(y2 - y1)
            gau_map = np.multiply(dy, np.transpose(dx))
            
            mask[0, y1:y2, x1:x2] = np.maximum(mask[0, y1:y2, x1:x2], gau_map)
            mask[1, y1:y2, x1:x2] = 1
            mask[2, c_y, c_x] = 1


            scale_map[0, c_y - self.radius:c_y + self.radius + 1, c_x - self.radius:c_x + self.radius + 1] = np.log(
                (y2-y1)+1e-10)  # log value of height
            scale_map[1, c_y - self.radius:c_y + self.radius + 1, c_x - self.radius:c_x + self.radius + 1] = np.log(
                (x2 - x1)+1e-10)  # log value of width

            scale_map[2, c_y - self.radius:c_y + self.radius + 1, c_x - self.radius:c_x + self.radius + 1] = 1

            offset_map[0, c_y, c_x] = (y1 + y2) / 2 - c_y - 0.5
            offset_map[1, c_y, c_x] = (x1 + x2) / 2 - c_x - 0.5
            offset_map[2, c_y, c_x] = 1

        return mask, scale_map, offset_map










