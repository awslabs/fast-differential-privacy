### Two Privacy Engines

FastDP provides two privacy engines to compute the private gradient: **hook-based** and **torch-extending**. These privacy engines are equivalent mathematically, though their applicability and computation efficiency can be different. We summarize the differences and note that some limitations can be overcome with more engineering efforts.

|                           | Hook-based (DP)                  | Torch-extending (DP) | Standard (non-DP)    |
|:----------------------------:|:-------------------------------:|:----------------:|:------------:| 
| Speed (1/time complexity)     | ~120%                            | ~150%            | 100% |
| Memory cost (space complexity)     | 100-130% | ~100%             | 100%         | 
| ZeRO distribution solution   | ✅ Supported              | ✅ Supported  | ✅ Supported |
| Most types of layers     | ✅ Supported (see below)                  | ✅ Supported (see below)   | ✅ Supported  | 
| Per-sample clipping styles     | ✅ Supported for all styles                   | Layer-wise style   |✅ Not needed  |
| Per-sample clipping functions  | ✅ Supported for all functions                | Automatic clipping    |✅ Not needed  |
| Modifying optimizers | Needed for `PrivacyEngine`; not needed for ZeRO    | ✅ Not needed   | ✅ Not needed |
| Private gradient stored in        | `param.private_grad`                    | `param.grad`   | `param.grad`  |
| Fused kernel  | ✅ Supported                | Not supported    |✅ Supported  |
| Ghost differentiation (origin param)           | Supported on single GPU        | Not supported   | Not needed  |
| Recommended usage       | Single GPU or ZeRO   | General   | General  |

#### 1. Hook-based
Hook-based approach computes the private gradient with forward hooks (to store the activations) and backward hooks (to compute the per-sample gradient norms, to clip and to add noise). See [this tutorial for hooks](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html). This approach firstly computes the private gradient then overrides the non-DP gradient.

On single GPU or data parallelism (see `PrivacyEngine`), the hooks are backward module hooks, which are triggered before `param.grad` is computed; in ZeRO (see `PrivacyEngine_Distributed_Stage_2_and_3`), some backward tensor hooks are in place, which are triggered after `param.grad` has been computed.

#### 2. Torch-extending
Torch-extending approach computes the private gradient directly by re-writeing the model's back-propagation mechanism (see `PrivacyEngine_Distributed_extending`). See [this tutorial for extending torch modules](https://pytorch.org/docs/stable/notes/extending.html#extending-torch-nn). This approach overrides the non-DP modules as shown in `supported_differentially_private_layers.py`. Given that this approach does not modify the optimizers and the communication orchestra of distributed solutions, it is expected to be applicable generally. However, some slowdown may be observed as the extension is not implemented at C++ level.

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
[param.requires_grad_(False) for name, param in model.named_parameters() if '.bias' not in name]
```
Note that for two-phase DP training (e.g. appendix of [6] or DP continual training), one need to detach the first engine and attach a new engine to a new optimizer.

### References
[1] Goodfellow, Ian. "Efficient per-example gradient computations." arXiv preprint arXiv:1510.01799 (2015).

[2] Li, Xuechen, Florian Tramer, Percy Liang, and Tatsunori Hashimoto. "Large language models can be strong differentially private learners." arXiv preprint arXiv:2110.05679 (2021).

[3] Bu, Zhiqi, Jialin Mao, and Shiyun Xu. "Scalable and Efficient Training of Large Convolutional Neural Networks with Differential Privacy." arXiv preprint arXiv:2205.10683 (2022).

[4] Bu, Zhiqi, Yu-Xiang Wang, Sheng Zha, and George Karypis. "Differentially Private Optimization on Large Model at Small Cost." arXiv preprint arXiv:2210.00038 (2022).

[5] Yousefpour, Ashkan, Igor Shilov, Alexandre Sablayrolles, Davide Testuggine, Karthik Prasad, Mani Malek, John Nguyen et al. "Opacus: User-friendly differential privacy library in PyTorch." arXiv preprint arXiv:2109.12298 (2021).

[6] Bu, Zhiqi, Yu-Xiang Wang, Sheng Zha, and George Karypis. "Differentially Private Bias-Term only Fine-tuning of Foundation Models." arXiv preprint arXiv:2210.00036 (2022).
