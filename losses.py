import torch
from torch import nn


class FocalLoss(nn.Module):

    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target, sigmoid=True):
        if sigmoid:  # clamp is important
            output = torch.clamp(output.sigmoid(), min=1e-4, max=1 - 1e-4)

        pos_index = target.eq(1).float()
        neg_index = target.lt(1).float()

        pos_loss = torch.pow(1 - output, self.alpha) * torch.log(output) * pos_index
        neg_loss = torch.pow(1 - target, self.beta) * torch.pow(output, self.alpha) * torch.log(1 - output) * neg_index

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        pos_num = pos_index.sum()
        loss = 0
        loss = loss - (pos_loss + neg_loss) / pos_num if pos_num > 0 else loss - neg_loss

        return loss


class RegL1Loss(nn.Module):

    def __init__(self):
        super(RegL1Loss, self).__init__()
        # 只统计包含目标的元素
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.eps = 1e-4

    def forward(self, output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, index: torch.Tensor):
        '''
        l1_loss from wh_loss or off_loss
        :param output: the output of model (batch,2,128,128)
        :param target: (batch,max_objs,2)   max_objs=128
        :param mask: (batch,max_objs)
        :param index: (batch,max_objs)
        :return:
        '''
        # index为1维索引,把output的特征图也转成1维
        batch = output.size(0)
        output = output.view(batch, 2, -1).transpose(1, 2).contiguous()  # (batch,128*128,2)
        # torch.gather函数需要index的维度和output保持一致(提取那一维除外)
        index = index.unsqueeze(2).expand(batch, -1, 2)  # (batch,max_objs,2)
        # get the target number
        pos_num = mask.sum()
        # 提取target对应feature值
        output = torch.gather(output, 1, index)  # (batch,max_objs,2)
        # 使用mask对无目标的output进行mask,先扩展维度
        mask = mask.unsqueeze(2).expand_as(output).float()  # (batch,max_objs,2)

        loss = self.l1_loss(output * mask, target * mask)
        loss = loss / (pos_num + self.eps)
        return loss
