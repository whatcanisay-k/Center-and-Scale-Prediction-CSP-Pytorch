import torch
import cv2
import argparse
import numpy as np
from net.csp import Csp
#from utils.nms_wrapper import nms
from utils.nms.py_cpu_nms import py_cpu_nms
import time

parser = argparse.ArgumentParser(description='CSP Testing')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
args = parser.parse_args()

class Config(object):
    def __init__(self):
        self.train_size = (640, 1280)
        self.test_size = (1024, 2048)
        self.stride = 4
        self.radius = 2
        self.epoch_size = 300
        self.score_thres = 0.2
        self.nms_thres = 0.3



def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))

    model.load_state_dict(pretrained_dict)
    return model


def parse_det_offset(output, config):
    size, score, down, nms_thresh = config.test_size, config.score_thres, config.stride, config.nms_thres
    pos, scale, offset = output
    pos = np.squeeze(pos.data.cpu())
    height = scale.data.cpu()[0, 0, :, :].numpy()
    width = scale.data.cpu()[0, 1, :, :].numpy()

    offset_y = offset.data.cpu()[0, 0, :, :].numpy()
    offset_x = offset.data.cpu()[0, 1, :, :].numpy()
    y_c, x_c = np.where(pos > score)
    boxs = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            #w = 0.41 * h
            w = np.exp(width[y_c[i], x_c[i]]) * down
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = pos[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x + 0.5) * down - w / 2), max(0, (y_c[i] + o_y + 0.5) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, size[1]), min(y1 + h, size[0]), s])
        boxs = np.asarray(boxs, dtype=np.float32)
        #keep = nms(boxs, nms_thresh, True) # cpu nms
        keep = py_cpu_nms(boxs, nms_thresh)
        boxs = boxs[keep, :]
    return boxs

def test():

    config = Config()
    torch.set_grad_enabled(False)

    net = Csp('test')
    net = load_model(net, './weights/CSP_epoch_320.pth', args.cpu)

    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    net.eval()

    img = np.float32(cv2.imread('./frankfurt_000000_000294_leftImg8bit.png', cv2.IMREAD_COLOR))
    im2show = img.copy()
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)

    begin_time = time.time()
    output = net(img)
    print('forward time: {}'.format(time.time()-begin_time))

    begin_time = time.time()
    boxes= parse_det_offset(output, config)
    print('boxes time: {}'.format(time.time()-begin_time))

    for box in boxes:
        cv2.rectangle(im2show, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [0, 0, 255], 2)

    cv2.imwrite('./result.jpg', im2show)

if __name__ == '__main__':
    test()

