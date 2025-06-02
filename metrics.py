import math
import numpy as np

import torch


def cc_weighted(pred_up, fdm_up, region_map):
    B = pred_up.size(0)
    cost_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0]
        gm = fdm_up[b, 0]
        wm = region_map[b, 0]

        pm_w = pm * wm
        gm_w = gm * wm
        pm_mean = pm_w.mean()
        gm_mean = gm_w.mean()
        pm_c = pm_w - pm_mean
        gm_c = gm_w - gm_mean
        num = (pm_c * gm_c).sum()
        den = torch.sqrt((pm_c ** 2).sum() * (gm_c ** 2).sum() + 1e-8)
        c = num / (den + 1e-8)
        cost_i = 1. - c
        cost_sum += cost_i
    return cost_sum / B


def nss_weighted(pred_up, fix_720, region_map):
    B = pred_up.size(0)
    vals = []
    for b in range(B):
        pm = pred_up[b, 0]
        fm = fix_720[b, 0]
        wm = region_map[b, 0]

        pm_w = pm * wm
        fix_sum = fm.sum().item()
        if fix_sum < 1:
            vals.append(0.)
            continue
        mn = pm_w.mean()
        st = pm_w.std(unbiased=False) + 1e-8
        z = (pm_w - mn) / st
        val = float((z * fm).sum().item() / (fix_sum + 1e-8))
        vals.append(val)
    return float(np.mean(vals))


def kl_weighted(pred_up, fdm_up, region_map):
    B = pred_up.size(0)
    kl_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0].clone()
        gm = fdm_up[b, 0].clone()
        wm = region_map[b, 0]

        pm_w = pm * wm
        gm_w = gm * wm
        psum = pm_w.sum().item()
        gsum = gm_w.sum().item()
        if psum < 1e-8 or gsum < 1e-8:
            continue
        pm_w /= psum
        gm_w /= gsum
        ratio = (gm_w + 1e-8) / (pm_w + 1e-8)
        kl_val = (gm_w * torch.log(ratio)).sum().item()
        kl_sum += kl_val
    return kl_sum / B


def cc_2(pred_up, fdm_up):
    B = pred_up.size(0)
    cost_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0]
        gm = fdm_up[b, 0]
        pm_mean = pm.mean()
        gm_mean = gm.mean()
        pm_c = pm - pm_mean
        gm_c = gm - gm_mean
        num = (pm_c * gm_c).sum()
        den = torch.sqrt((pm_c ** 2).sum() * (gm_c ** 2).sum() + 1e-8)
        cost_i = 1. - (num / (den + 1e-8))
        cost_sum += cost_i
    return cost_sum / B


def nss_2(pred_up, fix_720):
    B = pred_up.size(0)
    vals = []
    for b in range(B):
        pm = pred_up[b, 0]
        fm = fix_720[b, 0]
        s = fm.sum().item()
        if s < 1:
            vals.append(0.)
            continue
        mn = pm.mean()
        st = pm.std(unbiased=False) + 1e-8
        pmZ = (pm - mn) / st
        val = (pmZ * fm).sum().item() / (s + 1e-8)
        vals.append(val)
    return float(np.mean(vals))


def kl_2(pred_up, fdm_up):
    B = pred_up.size(0)
    kl_sum = 0.
    for b in range(B):
        pm = pred_up[b, 0].clone()
        gm = fdm_up[b, 0].clone()
        psum = pm.sum().item()
        gsum = gm.sum().item()
        if psum < 1e-8 or gsum < 1e-8:
            continue
        pm /= psum
        gm /= gsum
        ratio = (gm + 1e-8) / (pm + 1e-8)
        kl_val = (gm * torch.log(ratio)).sum().item()
        kl_sum += kl_val
    return kl_sum / B


def discretize_gt(gt):
    out = np.zeros_like(gt, dtype=np.float32)
    out[gt > 0] = 1.0
    return out


def nss(s_map, gt):
    gt = discretize_gt(gt)
    s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-8)
    x, y = np.where(gt == 1)
    if len(x) == 0:
        return 0.0
    vals = [s_map_norm[xx, yy] for (xx, yy) in zip(x, y)]
    return float(np.mean(vals))


def cc(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-8)
    gt_norm = (gt - np.mean(gt)) / (np.std(gt) + 1e-8)
    num = (s_map_norm * gt_norm).sum()
    den = math.sqrt((s_map_norm ** 2).sum() * (gt_norm ** 2).sum() + 1e-8)
    return float(num / (den + 1e-8))


def kldiv(s_map, gt):
    eps = 1e-16
    s_sum = s_map.sum()
    g_sum = gt.sum()
    if s_sum < eps or g_sum < eps:
        return 0.0
    p = s_map / (s_sum + eps)
    q = gt / (g_sum + eps)
    ratio = (q + eps) / (p + eps)
    return float((q * np.log(ratio)).sum())


def total_variation_loss(pred_36x64):
    dx = torch.abs(pred_36x64[:, :, 1:, :] - pred_36x64[:, :, :-1, :])
    dy = torch.abs(pred_36x64[:, :, :, 1:] - pred_36x64[:, :, :, :-1])
    return dx.mean() + dy.mean()
