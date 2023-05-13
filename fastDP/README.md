## Notes for privacy engine

### Supported Modules

Our privacy engine supports the commonly used modules that constitute most models, with possibly two methods to compute the per-sample gradient norm:
* nn.Linear (GhostClip & Grad Instantiation)
* nn.LayerNorm (Grad Instantiation)
* nn.GroupNorm (Grad Instantiation)
* nn.InstanceNorm (Grad Instantiation)
* nn.Embedding (GhostClip)
* nn.Conv1d (GhostClip & Grad Instantiation)
* nn.Conv2d (GhostClip & Grad Instantiation)
* nn.Conv3d (GhostClip & Grad Instantiation)

Frozen (e.g. `nn.Linear` with `requires_grad=False`) and non-trainable (e.g. `nn.ReLU`, `nn.Tanh`, `nn.MaxPool2d`) modules are also supported.

Note GhostClip stands for ghost clipping [1][2][3], that computes the gradient norms without creating and storing the gradients. Grad Instantiation stands for per-sample gradient instantiation [5], that generates the per-sample gradients and then computes their norms. Note that Grad Instantiation can be inefficient for large models and GhostClip can be inefficient for high-dimensional data. Therefore we allow to choose the method at different layers (known as the hybrid algorithms by [3][4]) for modules that support both methods.

### Arguments
* `module`: The model that to be optimized with differential privacy.
* `batch_size`: Logical batch size that determines the convergence and accuracy.
* `sample_size`: Number of training samples.
* `target_epsilon`: Target privacy budget ε.
* `target_delta`: Target privacy budget δ, should be smaller than 1/sample_size.
* `max_grad_norm`: Per-sample gradient clipping threshold, default to 1. No need to tune if `clipping_fn="automatic"`.
* `epochs`: Number of epochs. Not needed if `noise_multiplier` is provided.
* `noise_multiplier`: Level of independent Gaussian noise into the gradient. This can be automatically computed by different `accounting_mode` if `target_epsilon, batch_size, sample_size, epochs` are provided.
* `accounting_mode`: Privacy accounting theory to use, one of "rdp" (default), "glw", "all".
* `named_params`: Specifies which parameters to optimize with differential privacy.
* `clipping_mode`: Per-sample gradient clipping mode, one of 'ghost', 'MixGhostClip', 'MixOpt' (default) from [4]. Note different clipping modes, including Opacus [5], GhostClip [2] and Mixed GhostClip [3], give the same convergence and accuracy though at significantly different time/space complexity.
* `clipping_fn`: Per-sample gradient clipping function to use; one of "automatic" (default, [Bu et al., 2022](https://arxiv.org/pdf/2206.07136.pdf)), "Abadi" [(Abadi et al., 2016)](https://arxiv.org/pdf/1607.00133.pdf) , "global" [(Bu et al., 2021)](https://arxiv.org/pdf/2106.07830.pdf).
* `clipping_style`: Per-sample gradient clipping style to use; one of `all-layer` (flat clipping), `layer-wise` (each layer is a block, including both weight and bias parameters), `param-wise` (each parameter is a block), or a list of layer names (general block-wise clipping).
* `--origin_params`: Origin parameters for the ghost differentiation trick from [Bu et al. Appendix D.3](https://arxiv.org/pdf/2210.00038.pdf). Default is `None` (not using the trick). To enjoy the acceleration from the trick, set to each model's first trainable layer's parameters. For example, in text classification with RoBERTa, set `origin_params=["_embeddings"]`; in text generation with GPT2, set `origin_params=["wte","wpe"]`; in image classification with BEiT, set `origin_params=["patch_embed.proj.bias"]`. This trick gives about 8/6=1.666 speedup at no memory overhead.

### Usage
Our privacy engine uses Pytorch [forward and backward hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html) to clip per-sample gradients and to add noises. To privately train models, attach the privacy engine to any optimizers from [torch.optim](https://pytorch.org/docs/stable/optim.html), which accumulates the sum of clipped per-sample gradients into `.grad` during backward propagation and additionally inject noises by `step`.

To conduct DP bias-term fine-tuning (DP-BiTFiT [6]), simply freeze all non-bias terms:
```python
for name, param in model.named_parameters():
    if '.bias' in name:
        param.requires_grad_(False)
```
Note that for two-phase DP training (Section 4.4, [6]), one need to detach the first engine and attach a new engine to a new optimizer. See `image_classification/CIFAR_TIMM_2phase.py` for details.

### References
[1] Goodfellow, Ian. "Efficient per-example gradient computations." arXiv preprint arXiv:1510.01799 (2015).
[2] Li, Xuechen, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. "Large language models can be strong differentially private learners." arXiv preprint arXiv:2110.05679 (2021).
[3] Bu, Zhiqi, Jialin Mao, and Shiyun Xu. "Scalable and Efficient Training of Large Convolutional Neural Networks with Differential Privacy." arXiv preprint arXiv:2205.10683 (2022).
[4] Bu, Zhiqi, Yu-Xiang Wang, Sheng Zha, and George Karypis. "Differentially Private Optimization on Large Model at Small Cost." arXiv preprint arXiv:2210.00038 (2022).
[5] Yousefpour, Ashkan, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek, John Nguyen et al. "Opacus: User-friendly differential privacy library in PyTorch." arXiv preprint arXiv:2109.12298 (2021).
[6] Bu, Zhiqi, Yu-Xiang Wang, Sheng Zha, and George Karypis. "Differentially Private Bias-Term only Fine-tuning of Foundation Models." arXiv preprint arXiv:2210.00036 (2022).