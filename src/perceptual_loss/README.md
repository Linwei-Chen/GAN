# Style Transfer Project usage


## 1. training
```
cd /src/style_transfer_perceptual_loss/
python train_style_transfer.py \
--image_dataset /media/charles/STORAGE/dataset/VOC2012Image \ # a dir has some dir which containing natural image
--save ../../style_transfer \ # a dir for saving model
--batch_size 14 \
--display 1 \
--epochs 10 \
--crop_size 256 \
--aug crop \ random crop
--style_image /media/charles/STORAGE/code/20191107GAN/src/style_transfer_perceptual_loss/style_imgs/TheStarryNight.jpg # target style image
```

## 2. transferring

```
python image_transfer.py 
--img_dir /dir_containing_your_pic \ # a dir has some dir which containing natural 
--style_image /media/charles/STORAGE/code/20191107GAN/src/style_transfer_perceptual_loss/style_imgs/TheStarryNight.jpg # you should train on this image first
```

or use bulit-in style:
```
python image_transfer.py 
--img_dir /dir_containing_your_pic \ # a dir has some dir which containing natural 
--style TheStarryNight
``` 