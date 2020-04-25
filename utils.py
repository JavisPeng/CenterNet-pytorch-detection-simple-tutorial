import torch
import numpy as np
from torch.nn import functional as F


# =================================-----FOR DATASET------============================

def gaussian2D(shape, sigma=1):
    '''
    对输入shape(radius,radius)生产一个高斯核
    :param shape: (diameter,diameter)
    :param sigma:
    :return: (radius*2+1,radius*2+1)
    '''
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    # 过于小的数设置为0
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    '''
    在heatmap上对中心点center半径为radius画高斯分布
    :param heatmap: 特征图(128*128)
    :param center: (x,y)
    :param radius: 半径
    :param k:
    :return:
    '''
    # 从中心开始扩展,长度为2 * radius + 1,文章说是sigma=radius/3,
    # 这里也解决了radius=0，导致sigma除数为0的问题
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center[0], center[1]

    height, width = heatmap.shape[0:2]

    # 越界处理
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    # 对齐处理
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    '''
    求高斯半径
    方法来自于CornerNet:https://arxiv.org/pdf/1808.01244.pdf
    原理就是对三种情况(1内扩1外扩,2内扩,2外扩)解一元二次方程:https://github.com/princeton-vl/CornerNet/issues/110
    :param det_size: bbox在特征图的大小(h,w)
    :param min_overlap: 最小的IOU
    :return: 最小的半径，其保证iou>=min_overlap
    '''
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


# =================================-----FOR TEST-----============================

def hm_topk(hm, k):
    # 使用max_pool获取峰点
    batch, cls, h, w = hm.size()
    out = F.max_pool2d(hm, 3, 1, 1)
    keep_max = (out == hm).float()
    hm = keep_max * hm
    # 在heatmap中取每个类别的topk(hm经sigmoid后) topk_indexs的值在0~h*w
    topk_scores, topk_indexs = hm.view(batch, cls, -1).topk(k)  # (batch,cls,k)
    # 所有类别取得分最高的topk
    topk_scores, topk_ind = topk_scores.view(batch, -1).topk(k)  # (batch,k)
    # 获取得分最高的类别topk,topk_scores每个类别有得分最高的k个，topk_ind除k取下整即为class
    topk_cls = topk_ind // k
    # 若topk_indexs.size=(batch,cls*k),topk_indexs[topk_ind]即为最终的index
    topk_indexs = topk_indexs.view(batch, -1).gather(1, topk_ind)
    # 获取所有类别中最高得分topk_indexs对应的横纵坐标,即一维转二维
    topk_ys, topk_xs = topk_indexs // w, topk_indexs % w
    return topk_scores, topk_indexs, topk_cls, topk_xs, topk_ys


def heatmap_bbox(hm, wh, reg, k=100):
    scores, indexs, cls, xs, ys = hm_topk(hm.sigmoid_(), k)
    batch = reg.size(0)
    # 先转置便于取关键点对应的2个偏移量
    reg = reg.view(batch, 2, -1).transpose(2, 1).contiguous()  # (batch,w*h,2)
    reg_indexs = indexs.unsqueeze(2).expand(batch, -1, 2)  # (batch,k,2)
    reg = reg.gather(1, reg_indexs)  # (batch,k,2)
    xs = xs.float() + reg[:, :, 0]
    ys = ys.float() + reg[:, :, 1]
    # wh via reg_indexs
    wh = wh.view(batch, 2, -1).transpose(2, 1).contiguous().gather(1, reg_indexs)  # ((batch,k,2)
    # bbox via xs and wh
    bbox = xs - wh[:, :, 0] / 2, ys - wh[:, :, 1] / 2, xs + wh[:, :, 0] / 2, ys + wh[:, :, 1] / 2
    bbox = torch.stack(bbox, -1)  # (batch,k,4)
    return bbox, cls, scores


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def soft_nms(box_scores, score_threshold=0.5, sigma=0.5, top_k=-1):
    """Soft NMS implementation.

    References:
        https://arxiv.org/abs/1704.04503
        https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/cython_nms.pyx
        https://oldpan.me/archives/write-hard-nms-c

    Args:
        box_scores (N, 6): boxes in corner-form and probabilities. [x1,y1,x2,y2,cls,score]
        score_threshold: boxes with scores less than value are not considered.
        sigma: the parameter in score re-computation.
            scores[i] = scores[i] * exp(-(iou_i)^2 / simga)
        top_k: keep top_k results. If k <= 0, keep all the results.
    Returns:
         picked_box_scores (K, 5): results of NMS.
    """
    picked_box_scores = []
    while box_scores.size(0) > 0:
        max_score_index = torch.argmax(box_scores[:, -1])
        cur_box_prob = box_scores[max_score_index, :].clone().detach()
        picked_box_scores.append(cur_box_prob)
        if len(picked_box_scores) == top_k > 0 or box_scores.size(0) == 1:
            break
        cur_box = cur_box_prob[:-2]
        box_scores[max_score_index, :] = box_scores[-1, :]
        box_scores = box_scores[:-1, :]
        ious = iou_of(cur_box.unsqueeze(0), box_scores[:, :-2])

        box_scores[:, -1] = box_scores[:, -1] * torch.exp(-(ious * ious) / sigma)

        box_scores = box_scores[box_scores[:, -1] > score_threshold, :]
    if len(picked_box_scores) > 0:
        return torch.stack(picked_box_scores)
    else:
        return torch.tensor([])
