# CenterNet-pytorch-detection-simple-tutorial

## environment
```
tqdm>=4.32.2
numpy>=1.15.0
torch>=1.0
torchvision>=0.2.1
Pillow>=7.1.1
```

## data
SeaShips dataset：7000 images，1920x1080 size，6 claases
http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips%287000%29.zip

## data preprocess
dowload the SeaShips dataset and unzip 
```shell
git clone https://github.com/JavisPeng/CenterNet-pytorch-detection-simple-tutorial.git
cd CenterNet-pytorch-detection-simple-tutorial/
mkdir VOC2007 & cd VOC2007
wget http://www.lmars.whu.edu.cn/prof_web/shaozhenfeng/datasets/SeaShips%287000%29.zip
unzip SeaShips(7000).zip
python voc.py
```

## train
train a model from scrtch
```python
python main.py train
```

## test
predict a image
```
python main.py test --ckpt w.pth --test_img_path VOC2007/JPEGImages/000001.jpg
```

a output demo
![test_image](https://img-blog.csdnimg.cn/20200424211506362.jpg)

## reference
https://github.com/xingyizhou/CenterNet
