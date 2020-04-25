import numpy as np
import utils
from torch.utils.data import Dataset
from PIL import Image


class CTDataset(Dataset):
    def __init__(self, opt, data, transform=None):
        '''
        数据集构建
        :param opt: 配置参数
        :param data: [(img_path,[(bbox1,cls1),(bbox2,cls2),]) bbox(左上右下)
        :param transform:
        '''

        self.images = data
        self.opt = opt
        self.transform = transform

    def __getitem__(self, index):
        img_path, list_bbox_cls = self.images[index]
        img = Image.open(img_path)
        real_w, real_h = img.size
        if self.transform: img = self.transform(img)
        heatmap_size = self.opt.input_size // self.opt.down_ratio
        # heatmap
        hm = np.zeros((self.opt.num_classes, heatmap_size, heatmap_size), dtype=np.float32)
        # withd and hight
        wh = np.zeros((self.opt.max_objs, 2), dtype=np.float32)
        # regression
        reg = np.zeros((self.opt.max_objs, 2), dtype=np.float32)
        # index in 1D heatmap
        ind = np.zeros((self.opt.max_objs), dtype=np.int)
        # 1=there is a target in the list 0=there is not
        reg_mask = np.zeros((self.opt.max_objs), dtype=np.uint8)

        # get the absolute ratio
        w_ratio = self.opt.input_size / real_w / self.opt.down_ratio
        h_ratio = self.opt.input_size / real_h / self.opt.down_ratio

        for i, (bbox, cls) in enumerate(list_bbox_cls):
            # original bbox size -> heatmap bbox size
            bbox = bbox[0] * w_ratio, bbox[1] * h_ratio, bbox[2] * w_ratio, bbox[3] * h_ratio
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            # center point(x,y)
            center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            center_int = center.astype(np.int)
            reg[i] = center - center_int
            wh[i] = 1. * width, 1. * height
            reg_mask[i] = 1
            ind[i] = center_int[1] * heatmap_size + center[0]
            radius = utils.gaussian_radius((height, width))
            #半径保证为整数
            radius = max(0, int(radius))
            utils.draw_umich_gaussian(hm[cls], center_int, radius)
        return (img, hm, wh, reg, ind, reg_mask)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    pass
