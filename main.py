import torch, tqdm
from models import cnet
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CTDataset
from losses import FocalLoss, RegL1Loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(41)


def train_epoch(epoch, model, dl, optimizer, cerition_hm, cerition_wh, cerition_reg):
    model.train()
    loss_meter, it = 0, 0
    bar = tqdm.tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    for item in bar:
        item = [x.to(device) for x in item]
        img, hm, wh, reg, ind, reg_mask = item
        optimizer.zero_grad()
        out_hm, out_wh, out_reg = model(img)
        hm_loss = cerition_hm(out_hm, hm)
        wh_loss = cerition_wh(out_wh, wh, reg_mask, ind)
        reg_loss = cerition_reg(out_reg, reg, reg_mask, ind)
        loss = hm_loss + 0.1 * wh_loss + reg_loss
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        bar.set_postfix(hm_loss=hm_loss.item(), wh_loss=wh_loss.item(), reg_loss=reg_loss.item(), loss=loss.item())
        it += 1
    return loss_meter / it


@torch.no_grad()
def val_epoch(model, dl, cerition_hm, cerition_wh, cerition_reg):
    model.eval()
    loss_meter, it = 0, 0
    for item in dl:
        item = [x.to(device) for x in item]
        img, hm, wh, reg, ind, reg_mask = item
        out_hm, out_wh, out_reg = model(img)
        hm_loss = cerition_hm(out_hm, hm)
        wh_loss = cerition_wh(out_wh, wh, reg_mask, ind)
        reg_loss = cerition_reg(out_reg, reg, reg_mask, ind)
        loss = hm_loss + 0.1 * wh_loss + reg_loss
        loss_meter += loss.item()
        it += 1
    return loss_meter / it


def train(opt):
    # model
    model = cnet(nb_res=opt.resnet_num, num_classes=opt.num_classes)
    model = model.to(device)

    transform_train = transforms.Compose([
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 数据文件通过voc.py生成
    dic_data = torch.load('data.pth')

    train_dataset = CTDataset(opt=opt, data=dic_data['train'], transform=transform_train)
    val_dataset = CTDataset(opt=opt, data=dic_data['val'], transform=transform_train)
    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    val_dl = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)

    cerition_hm = FocalLoss()
    cerition_wh = RegL1Loss()
    cerition_reg = RegL1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    min_loss, best_epoch = 1e7, 1
    for epoch in range(1, opt.max_epoch + 1):
        train_loss = train_epoch(epoch, model, train_dl, optimizer, cerition_hm, cerition_wh, cerition_reg)
        val_loss = val_epoch(model, val_dl, cerition_hm, cerition_wh, cerition_reg)
        print("Epoch%02d train_loss:%0.3e val_loss:%0.3e min_loss:%0.3e(%02d)" % (
            epoch, train_loss, val_loss, min_loss, best_epoch))
        if min_loss > val_loss:
            min_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), opt.ckpt)


@torch.no_grad()
def test(opt):
    model = cnet(nb_res=opt.resnet_num, num_classes=opt.num_classes)
    model.load_state_dict(torch.load(opt.ckpt, map_location='cpu'))
    model = model.to(device)

    transform_x = transforms.Compose([
        transforms.Resize((opt.input_size, opt.input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    import os
    from PIL import Image, ImageDraw
    if not os.path.exists(opt.output_dir): os.mkdir(opt.output_dir)
    img_name = os.path.basename(opt.test_img_path)
    img0 = Image.open(opt.test_img_path)
    real_w, real_h = img0.size
    img = transform_x(img0)
    img = torch.unsqueeze(img, 0).to(device)
    out_hm, out_wh, out_reg = model(img)
    import utils
    bbox, cls, scores = utils.heatmap_bbox(out_hm, out_wh, out_reg, opt.topk)
    w_ratio = real_w * opt.down_ratio / opt.input_size
    h_ratio = real_h * opt.down_ratio / opt.input_size
    # 同一维度和类型，便于cat
    cls = cls.unsqueeze(-1).float()
    scores = scores.unsqueeze(-1)
    #只测试一张图片batch=1，去掉该维度
    bbox_cls_score = torch.cat([bbox, cls, scores], dim=-1).squeeze()
    #使用soft_nms过滤掉不同类别在同一个关键点位置的情况
    #bbox_cls_score = utils.soft_nms(bbox_cls_score, score_threshold=opt.threshold, top_k=opt.topk)
    bbox_cls_score = bbox_cls_score.cpu().numpy()
    for bcs in bbox_cls_score:
        box, cls, score = bcs[:4], int(bcs[4]), bcs[-1]
        print(box, cls, score)
        box = box[0] * w_ratio, box[1] * h_ratio, box[2] * w_ratio, box[3] * h_ratio
        draw = ImageDraw.Draw(img0)
        draw.rectangle(box, outline='blue')
        draw.text((box[0], box[1] - 10), "(%d,%0.3f)" % (cls, score), fill='blue')
    img0.save(os.path.join(opt.output_dir, img_name))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=['train', 'test'], help="train | test")
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number workers in dataloader')
    parser.add_argument('--max_epoch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--resnet_num', type=int, default=18, choices=[18, 34, 50, 101, 152],
                        help='resnet numner in [18,34,50,101,152]')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--input_size', type=int, default=512, help='image input size')
    parser.add_argument('--max_objs', type=int, default=16, help='max object number in a picture')
    parser.add_argument('--topk', type=int, default=4, help='topk in target')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for nms,default is 0.5')
    parser.add_argument('--down_ratio', type=int, default=4, help='downsample ratio')
    parser.add_argument('--ckpt', type=str, default='w.pth', help='the path of model weight')
    parser.add_argument('--test_img_path', type=str, default='VOC2007/JPEGImages/000019.jpg',
                        help='test image path')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    opt = parser.parse_args()

    if opt.method == "train":
        train(opt)
    elif opt.method == "test":
        test(opt)
