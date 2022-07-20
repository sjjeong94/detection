import torch
import torch.nn as nn


class IOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, gt):
        o_l, o_t, o_r, o_b = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        g_l, g_t, g_r, g_b = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

        filt = g_l > 0
        o_l, o_t, o_r, o_b = o_l[filt], o_t[filt], o_r[filt], o_b[filt]
        g_l, g_t, g_r, g_b = g_l[filt], g_t[filt], g_r[filt], g_b[filt]

        o_a = (o_t + o_b) * (o_l + o_r)
        g_a = (g_t + g_b) * (g_l + g_r)

        iw = torch.minimum(o_l, g_l) + torch.minimum(o_r, g_r)
        ih = torch.minimum(o_t, g_t) + torch.minimum(o_b, g_b)
        inter = iw * ih
        union = o_a + g_a - inter
        iou = inter / (union + 1e-9)
        #loss = -torch.log(iou)
        loss = 1 - iou
        return loss.mean()


class GIOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, gt):
        o_l, o_t, o_r, o_b = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        g_l, g_t, g_r, g_b = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

        filt = g_l > 0
        o_l, o_t, o_r, o_b = o_l[filt], o_t[filt], o_r[filt], o_b[filt]
        g_l, g_t, g_r, g_b = g_l[filt], g_t[filt], g_r[filt], g_b[filt]

        o_a = (o_t + o_b) * (o_l + o_r)
        g_a = (g_t + g_b) * (g_l + g_r)

        iw = torch.minimum(o_l, g_l) + torch.minimum(o_r, g_r)
        ih = torch.minimum(o_t, g_t) + torch.minimum(o_b, g_b)

        inter = iw * ih
        union = o_a + g_a - inter
        iou = inter / (union + 1e-9)

        cw = torch.maximum(o_l, g_l) + torch.maximum(o_r, g_r)
        ch = torch.maximum(o_t, g_t) + torch.maximum(o_b, g_b)
        c_area = cw * ch

        giou = iou - (c_area - union) / (c_area + 1e-9)
        loss = 1 - giou

        return loss.mean()
