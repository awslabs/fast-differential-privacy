## Image classification with convolutional neural networks and vision transformers

We provide the scripts to implement DP optimization on CIFAR10, CIFAR100, SVHN, ImageNet, CelebA, Places365, INaturalist, and other datasets, using the models (CNN and ViT) from [TIMM](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models). Supported models include VGG, ResNet, Wide ResNet, ViT, CrossViT, BEiT, DEiT, ... 

### CIFAR10/CIFAR100
```plaintext
python -m CIFAR_TIMM --model vit_large_patch16_224 --origin_params 'patch_embed.proj.bias' --clipping_mode BK-MixOpt --cifar_data CIFAR10
```

The script by default uses (hybrid) book-keeping by [Differentially Private Optimization on Large Model at Small Cost](https://arxiv.org/pdf/2210.00038.pdf) for the DP full fine-tuning. Gradient accumulation is used so that larger physical batch size allows faster training at heavier memory burden, but the accuracy is not affected. This script achieves state-of-the-art accuracy with BEiT-large and ViT-large under 7 min per epoch on one A100 GPU (40GB). Notice that `--origin_params 'patch_embed.proj.bias'` specifically accelerates ViT through the ghost differentiation trick.

Arguments:

* `--cifar_data`: Whether to train on CIFAR10 (default) or CIFAR100 datasets.

* `--epsilon`: Target privacy spending, default is 2.

* `--clipping_mode`: Which DP algorithm to implement per-sample gradient clipping; one of `nonDP` (non-private full fine-tuning), `BK-ghost` (base book-keeping), `BK-MixGhostClip`, `BK-MixOpt` (default), `BiTFiT` (DP bias-term fine-tuning) and `nonDPBiTFiT` (non-private BiTFiT). All BK algorithms are from [Bu et al., 2022](https://arxiv.org/pdf/2210.00038.pdf), and DP-BiTFiT is from [Bu et al., 2022](https://arxiv.org/pdf/2210.00036.pdf).

* `--model`: The pretrained model from TIMM, check the full list by `timm.list_models(pretrained=True)`.

* `--origin_params`: Origin parameters for the ghost differentiation trick from [Bu et al. Appendix D.3](https://arxiv.org/pdf/2210.00038.pdf). Default is `None` (not using the trick). To enjoy the acceleration from the trick, set to each model's first trainable layer's parameters.

* `--dimension`: Dimension of images, default is 224, i.e. the image is resized to 224*224.

* `--lr`: Learning rate, default is 0.0005. Note BiTFiT learning rate should be larger than full fine-tuning's.

* `--mini_bs` : Physical batch size for gradient accumulation that determines memory and speed, but not accuracy, default is 50.

* `--bs` : Logical batch size that determines the convergence and accuracy, should be multiple of `physical_batch_size`; default is 1000.

* `--epochs`: Number of epochs, default is 3.

* `--clipping_style`: Which group-wise per-sample gradient clipping style to use. This argument takes one of `all-layer` (flat clipping), `layer-wise` (each layer is a group, including both weight and bias parameters), `param-wise` (each parameter is a group), or a list of layer names (general group-wise clipping). For example, a uniform 3-group clipping can be implemented using 
```plaintext
python -m CIFAR_TIMM --model vit_base_patch16_224 --origin_params 'patch_embed.proj.bias' --clipping_style patch_embed.proj blocks.4.norm1 blocks.8.norm1 --cifar_data CIFAR10
```

### CelebA
Download dataset by `torchvision`, or from the [official host](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) with all .txt and `/img_align_celeba` in the same directory.
```plaintext
python -m CelebA_TIMM --model resnet18
```
Same arguments `[lr, epochs, bs, mini_bs, epsilon, clipping mode, model]` as the CIFAR example with one addition `--labels`. Default is `None` to train all 40 labels as multi-label/multi-task problem, otherwise train on label indices specified as a list. For example, label index 31 is 'Smiling' and label index 20 is 'Male'.

### General CV experiments
We provide a general script to experiment on many torchvision datasets, and we fix most of the arguments in the privacy engine.
```plaintext
python -m CV_TIMM --model vit_base_patch16_224 --dataset_name ImageNet
```

### Note
1. Vision models oftentimes have batch normalization layers, which violate the DP guarantee (see [Opacus](https://opacus.ai/tutorials/guide_to_module_validator) for the reason). A common solution is to replace with group/layer/instance normalization, and this can be easily fixed by Opacus>=v1.0: `model=ModuleValidator.fix(model)`.

2. To reproduce DP image classification and compare with other packages, we refer to [private-vision](https://github.com/woodyx218/private_vision) (covering GhostClip, MixGhostClip, Opacus-like optimization) and [Opacus](https://github.com/pytorch/opacus). Different packages and clipping modes should produce the same accuracy. Note that training more epochs with larger noise usually gives better accuracy.

3. Generally speaking, GhostClip is inefficient for large image (try 512X512 image with resnet18), Opacus is inefficient for large model (try 224X224 image with BEiT-large). Hence we improve on mixed ghost norm from [Bu et al.](https://arxiv.org/abs/2205.10683) to use GhostClip or Opacus at different layers.
