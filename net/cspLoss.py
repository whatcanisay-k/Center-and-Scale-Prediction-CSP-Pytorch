import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CspLoss(nn.Module):
    def __init__(self):
        super(CspLoss, self).__init__()
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')



    def forward(self, out, center_gt, scale_gt, offset_gt):
        center_out, sacle_hw, offset_pred= out[0].double(), out[1].double(), out[2].double()
        #center_out, sacle_hw= out[0].double(), out[1].double()
        sacle_h_out, sacle_w_out = sacle_hw[:, 0, :, :].unsqueeze(dim=1), sacle_hw[:, 1, :, :].unsqueeze(dim=1)

        pt = F.softmax(center_out, 1)
        logpt = torch.log(pt)
        #print(torch.max(pt))

        #pt_pos = (center_gt[:, 2, :, :]*pt[:, 1, :, :])
        #pt_neg = (center_gt[:, 1, :,  :]-center_gt[:, 2, :,  :])*pt[:, 0, :, :]
        #logpt_pos = torch.log(pt_pos+1e-10)
        #logpt_neg = torch.log(pt_neg+1e-10)

        #print(torch.max(logpt_pos))
        #print(torch.max(logpt_neg))


        positives = center_gt[:, 2, :, :]
        negatives = center_gt[:, 1, :, :] - center_gt[:, 2, :, :]

        fore_weight = positives * (1.0 - pt[:, 1, :, :]) ** 2*torch.log(pt[:, 1, :, :])
        back_weight = negatives * ((1.0 - center_gt[:, 0, :, :]) ** 4) * (1-pt[:, 0, :, :]) ** 2 * torch.log(pt[:, 0, :, :])

        focal_weight = fore_weight + back_weight
        assigned_box = torch.sum(center_gt[:, 2, :, :])

        center_loss = -1* torch.sum(focal_weight) / max(1.0, assigned_box)
        
        l1_loss_h = scale_gt[:, 2, :, :]*self.smoothl1(sacle_h_out[:, 0, :, :], scale_gt[:, 0, :, :])

        reg_loss_h = torch.sum(l1_loss_h) / max(1.0, torch.sum(scale_gt[:, 2, :, :]))

        l1_loss_w = scale_gt[:, 2, :, :]*self.smoothl1(sacle_w_out[:, 0, :, :], scale_gt[:, 1, :, :])

        reg_loss_w = torch.sum(l1_loss_w) / max(1.0, torch.sum(scale_gt[:, 2, :, :]))

        l1_loss_offset = offset_gt[:, 2, :, :].unsqueeze(dim=1)*self.smoothl1(offset_pred, offset_gt[:, :2, :, :])
        offset_loss = torch.sum(l1_loss_offset) / max(1.0, torch.sum(offset_gt[:, 2, :, :]))


        return center_loss, reg_loss_h, reg_loss_w, offset_loss


        


