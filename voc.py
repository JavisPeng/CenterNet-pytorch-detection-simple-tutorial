import os, torch
import xml.etree.ElementTree as ET

voc_root = 'VOC2007'
voc_Annotations = os.path.join(voc_root, 'Annotations')
voc_ImageSets = os.path.join(voc_root, 'ImageSets', 'Main')
voc_JPEGImages = os.path.join(voc_root, 'JPEGImages')


# https://blog.csdn.net/lingyunxianhe/article/details/81808539
def annot_box_loc(ann_path):  # AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=ann_path)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName = Object.find('name').text
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)  # -1
        x2 = int(BndBox.find('xmax').text)  # -1
        y2 = int(BndBox.find('ymax').text)  # -1
        BndBoxLoc = [x1, y1, x2, y2]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)  # 如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName] = [BndBoxLoc]  # 如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet


def get_classes_name():
    classes_name = set()
    for name in os.listdir(voc_Annotations):
        ann_path = os.path.join(voc_Annotations, name)
        tree = ET.ElementTree(file=ann_path)  # 打开文件，解析成一棵树型结构
        root = tree.getroot()  # 获取树型结构的根
        ObjectSet = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
        ObjBndBoxSet = {}  # 以目标类别为关键字，目标框为值组成的字典结构
        for Object in ObjectSet:
            ObjName = Object.find('name').text
            classes_name.add(ObjName)
    return classes_name


def get_data(set_path, idx2names):
    '''
    根据路径构建数据集
    :param txt: 数据集id列表
    :return: [img_path, [(bbox1, cls1), (bbox2, cls2), ]
    '''
    data = []
    for line in open(set_path):
        id = line.strip()
        img_path = os.path.join(voc_JPEGImages, "%s.jpg" % id)
        ann_path = os.path.join(voc_Annotations, "%s.xml" % id)
        dic_bbox = annot_box_loc(ann_path)
        bbox_cls = []
        for cls, list_bbox in dic_bbox.items():
            ix = idx2names[cls]
            for bbox in list_bbox:
                bbox_cls.append((bbox, ix))
        data.append([img_path, bbox_cls])
    return data



if __name__ == '__main__':
    train_set_path = os.path.join(voc_ImageSets, 'trainval.txt')
    val_set_path = os.path.join(voc_ImageSets, 'test.txt')
    classes_name = get_classes_name()

    idx2names = {name: i for i, name in enumerate(classes_name)}
    train = get_data(train_set_path, idx2names)
    val = get_data(val_set_path, idx2names)
    #前一半用来做测试，后一半用来做验证
    val=val[-len(val)//2:]
    data_info = {'classes_name': classes_name, 'train': train, 'val': val}
    torch.save(data_info, 'data.pth')
    print('classes_name', len(classes_name), classes_name)
    print('train', len(train), 'val', len(val))
    print(val[:10])



