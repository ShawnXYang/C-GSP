# Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks

The code for the paper '[Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks](https://arxiv.org/abs/2107.01809)' in ECCV 2022.

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- OS: Ubuntu 18.04.4
- GPU: Geforce 2080 Ti or Tesla P100
- Python: 3.6
- PyTorch: >= 1.6.0
- Torchvision: >= 0.6.0

## Running commands

### Training

- Please prepare the ImageNet set, and we adopt the ImageNet training set as our training data. 

- Below we provide running commands for training the conditional generator based on 8 different target classes from a previous same setting.


```python
python train.py --train_dir $DATA_PATH/ImageNet/train --model_type incv3 --eps 16 --batch_size 64 --start-epoch 0 --nz 16 --epochs 10 --label_flag 'N8'
```

```python
python train.py --train_dir $DATA_PATH/ImageNet/train --model_type res152 --eps 16 --batch_size 64 --start-epoch 0 --nz 16 --epochs 10 --label_flag 'N8'
```

Download pretrained adversarial generators [Generator-Inv3](https://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/C-GSP/model-inv3-epoch9.pth) and [Generator-Res152](https://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/C-GSP/model-res152-epoch9.pth) based on the setting of 8 different classes.

### Generating adversarial examples
Below we provide running commands for generating targeted adversarial examples on [ImageNet NeurIPS validation set](https://www.kaggle.com/c/nips-2017-non-targeted-adversarial-attack) (1k images):
```python
python eval_n8.py --data_dir data/ImageNet1k/ --model_type incv3 --eps 16 --load_path $SAVE_CHECKPOINT

```

### Testing
The above crafted adversarial examples can be directly used for testing different models in [torchvision](https://pytorch.org/vision/stable/models.html). Besides, you can also adopt Inception ResNet-V2 , ResNet-V2-152 in [Tensorflow Slim](https://github.com/tensorflow/models/tree/master/research/slim) or Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub> trained by [Ensemble Adversarial Training](https://git.dst.etit.tu-chemnitz.de/external/tf-models/-/tree/master/research/adv_imagenet_models)  in Tensorflow.

Below we provide running commands for testing our method against different black-box models: 
```python
python inference_n8.py --test_dir $IMAGES_DIR --model_t vgg16

```

### Basic Setting
Besides, we also provide all 20 pretrained adversarial generators [Generator-Inv3-1K.zip](https://ml.cs.tsinghua.edu.cn/~xiaoyang/downloads/C-GSP/Generator-Inv3-1K.zip) based on the basic setting of 1K classes.