"""Code for a privacy engine that enables deep learning with differential privacy

Design mostly based on Opacus and Private-transformers, and should work with 
most libraries such as huggingface, timm, torchvision, etc.
"""

import logging
import math
import types
from typing import Dict, Optional, Sequence, Union

import torch
from torch import nn

from . import transformers_support
from .accounting import accounting_manager
from torch.functional import F
import transformers
from .supported_differentially_private_layers import *


_DP_supported_layers = [nn.Embedding, 
    nn.Linear,
    nn.Conv2d, nn.LayerNorm, nn.GroupNorm, 
    transformers.pytorch_utils.Conv1D,
    transformers.models.llama.modeling_llama.LlamaRMSNorm
    ]
    
class PrivacyEngine_Distributed_extending(object):
    """Differentially-private optimization engine that works in Pytorch.

    Supports book-keeping (BK) algorithm -- base and hybrid variants, as described in arXiv:2210.00038
    Supports DP-BiTFiT (bias-term only fine-tuning, which does not use BK), as described in arXiv:2210.00036
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        batch_size: int,
        sample_size: int,
        epochs: Optional[Union[int, float]] = None,
        noise_multiplier: Optional[float] = None,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
        alphas: Sequence[float] = accounting_manager.DEFAULT_ALPHAS,
        numerical_stability_constant=None,
        accounting_mode="rdp",
        eps_error=0.05,
        clipping_fn='automatic',
        num_GPUs=1,
        torch_seed_is_fixed=True,
        grad_accum_steps=1,
        max_sequence_length=1,
        per_device_physical_batch_size = 8,
        per_sample_clip=True,
        **unused_kwargs,
    ):

        """Initialize the engine.

        Args:
            module: The PyTorch module for which per-sample gradient is required.
                Setting the `requires_grad` attribute of a parameter to False
                disables the per-sample gradient accumulation.
            batch_size: The expected size of Poisson-sampled batch, i.e., the lot size.
            sample_size: Size of dataset.
            epochs: The number of epochs for training.
            noise_multiplier: The extra multiplier for DP-SGD noise.
            target_epsilon: The target privacy spending.
                Only used to estimate the `noise_multiplier` if it is not set.
            target_delta: The target failure probability.
                Defaults to sample_size ** -1.1 if not set.!!!!!!!!!!!!
            alphas: The RDP orders for (ε, δ)-DP conversion. Useless if not accounting in RDP.
            named_params: Specifies which parameters need gradients;
                defaults to use parameters which require grad in module.
            numerical_stability_constant: Small constant to avoid division by 0 when clipping.
            accounting_mode: The method of accounting privacy. One of (`rdp`, `glw`, `all`).
                Meanings of shorthands:
                    - rdp: Account loss with RDP but perform conversion to approx-DP with a procedure defined in
                        "The Discrete Gaussian for Differential Privacy". https://arxiv.org/abs/2004.00010
                    - glw: Account loss by numerically composing tradeoff functions in f-DP; defined in
                        "Numerical composition of differential privacy". https://arxiv.org/abs/2106.02848
                    - all: Report loss with all methods listed above.
            eps_error: Error threshold for upper and lower bound in the GLW accounting procedure.
            clipping_mode: The clipping mode to use. One of 'ghost' (BK), 'MixGhostClip', 'MixOpt'.
            clipping_fn: Per-sample gradient clipping function to use. One of 'Abadi','automatic','global'
        """
        del unused_kwargs
        super(PrivacyEngine_Distributed_extending, self).__init__()

        if accounting_mode not in ("rdp", "all",'glw'):
            raise ValueError(f"Unknown accounting mode: {accounting_mode}. Expected one of 'rdp', 'all','glw'.")
        if epochs <= 0.0 and noise_multiplier is None:
            raise ValueError(f"Number of training epochs cannot be non-positive, but found epochs={epochs}")

        # Privacy parameters.
        sample_rate = batch_size / sample_size
        if target_delta is None:
            target_delta = 1 / sample_size
        if noise_multiplier is None:
            if target_epsilon is None or epochs is None:
                raise ValueError(
                    f"`target_epsilon` and `epochs` must be specified when `noise_multiplier` is `None`."
                )
            if accounting_mode in ("rdp", "all"):
                manager = accounting_manager.RDPManager(alphas=alphas)
            else:  # "glw"
                manager = accounting_manager.GLWManager(eps_error=eps_error)
            noise_multiplier = manager.compute_sigma(
                target_epsilon=target_epsilon, target_delta=target_delta, sample_rate=sample_rate, epochs=epochs,
            )

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.sample_rate = sample_rate

        self.epochs = epochs
        self.noise_multiplier = noise_multiplier
        self.effective_noise_multiplier = noise_multiplier / batch_size
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.alphas = alphas
        self.eps_error = eps_error
        self.accounting_mode = accounting_mode

        # Internals.
        self.steps = 0  # Tracks privacy spending.
        
        
        #-----
        def _supported_and_trainable(layer):
            if type(layer) in _DP_supported_layers and ((hasattr(layer,'weight') and hasattr(layer.weight,'requires_grad') and layer.weight.requires_grad) or (hasattr(layer,'bias') and hasattr(layer.bias,'requires_grad') and layer.bias.requires_grad)):
                return True
            return False
        # Record parameters.

        n_components=0
        self.n_layers=0

        for name,layer in module.named_modules():
            if _supported_and_trainable(layer):
                n_components+=sum([p.requires_grad for p in layer.parameters()])
                self.n_layers+=1
        print(f"Number of trainable components: {n_components}; Number of trainable layers: {self.n_layers}")


        transformers_support.forward_swapper(module=module)  # fix the position embeddings broadcast issue.

        #----- only support some layers
        for name,param in module.named_parameters():
            param.initially_requires_grad=param.requires_grad
            param.requires_grad=False
        for layer in module.modules():
            if type(layer) in _DP_supported_layers:
                [param.requires_grad_(True) for param in layer.parameters() if param.initially_requires_grad]

        trainable_param_dict={}
        DP_time_complexity={}
        DP_space_complexity={}
        nonDP_time_complexity={}
        nonDP_space_complexity={}
        for layer in module.modules():
            if hasattr(layer,'weight'):
                if hasattr(layer.weight,'ds_shape'): # in ZeRO3
                    layer.weight.shape=layer.weight.ds_shape
                    layer.weight.numels=layer.weight.ds_numel
                else:
                    layer.weight.numels=layer.weight.numel()
                if len(layer.weight.shape)==2:
                    B=per_device_physical_batch_size
                    d=layer.weight.shape[0]
                    p=layer.weight.shape[1]
                    if type(layer) in trainable_param_dict and layer.weight.requires_grad:
                        trainable_param_dict[type(layer)]+=layer.weight.numels
                        DP_time_complexity[type(layer)]+=B*(6*max_sequence_length*d*p+(2*max_sequence_length**2<p*d)*(2*max_sequence_length**2)*(p+d))
                        nonDP_time_complexity[type(layer)]+=B*(6*max_sequence_length*d*p)
                        DP_space_complexity[type(layer)]+=p*d+B*(min(2*(max_sequence_length**2),p*d)+max_sequence_length*(3*d+p))
                        nonDP_space_complexity[type(layer)]+=p*d+B*max_sequence_length*(3*d+p)
                    else:
                        trainable_param_dict[type(layer)]=layer.weight.numels
                        DP_time_complexity[type(layer)]=B*(6*max_sequence_length*d*p+(2*max_sequence_length**2<p*d)*(2*max_sequence_length**2)*(p+d))
                        nonDP_time_complexity[type(layer)]=B*(6*max_sequence_length*d*p)
                        DP_space_complexity[type(layer)]=p*d+B*(min(2*(max_sequence_length**2),p*d)+max_sequence_length*(3*d+p))
                        nonDP_space_complexity[type(layer)]=p*d+B*max_sequence_length*(3*d+p)
        #print(f"Trainable params: {sum(trainable_param_dict.values())}B.")
        print(f"DP/Standard(non-DP) time complexity : {sum(DP_time_complexity.values())/sum(nonDP_time_complexity.values())}.")
        for key in trainable_param_dict:
            print(f" ---- {key}: {trainable_param_dict[key]:.3e} param, relative speed {nonDP_time_complexity[key]/DP_time_complexity[key]}")
        print(f"DP/Standard(non-DP) space complexity : {sum(DP_space_complexity.values())/sum(nonDP_space_complexity.values())}.")
        for key in trainable_param_dict:
            print(f" ---- {key}: {trainable_param_dict[key]:.3e} param, relative speed {nonDP_time_complexity[key]/DP_time_complexity[key]}")

        replace_Embedding(module)
        replace_Linear(module)
        replace_Conv2d(module)
        replace_LayerNorm(module)
        replace_GroupNorm(module)
        replace_transformersConv1D(module)
        replace_llama_rsmnorm(module)
        
        
        self.module = module
        self.named_params = list(
            (name, param) for (name, param) in module.named_parameters() if param.requires_grad
        )
        self.num_params = sum(param.numel() for _, param in self.named_params)

        self._locked = False  # lock the part where noisy gradients is created (in `self.step`) if True.

        #-----
        for name,param in self.named_params:
            param.batch_size = self.batch_size
            param.n_layers = self.n_layers
            param.per_sample_clip = per_sample_clip
            
            if torch_seed_is_fixed == True:
                param.noise = self.noise_multiplier / num_GPUs / math.sqrt(grad_accum_steps)
            else:
                param.noise = self.noise_multiplier / math.sqrt(num_GPUs * grad_accum_steps)
                    
        print(f"Noise injected: {self.noise_multiplier} --> averaged by batch size: {self.effective_noise_multiplier}")

        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        # Make getting info easier.
        self.module.get_privacy_spent = types.MethodType(get_privacy_spent, self.module)

        self.module.privacy_engine = self


    def lock(self):
        """Run this after noisy clipped gradient is created to prevent tampering with it before parameter update."""
        self._locked = True

    def unlock(self):
        """Run this after parameter update to allow creation of noisy gradient for next step"""
        self._locked = False

    def get_privacy_spent(
        self,
        steps: Optional[int] = None,
        accounting_mode: Optional[str] = None,
        lenient=False
    ) -> Dict:
        if steps is None:
            steps = self.steps
        if accounting_mode is None:
            accounting_mode = self.accounting_mode

        privacy_results = {}  # Contains stats from all modes.
        if accounting_mode in ('all','rdp'):
            try:
                manager = accounting_manager.RDPManager(alphas=self.alphas)
                privacy_results.update(
                    manager.compute_epsilon(
                        sigma=self.noise_multiplier,
                        sample_rate=self.sample_rate,
                        target_delta=self.target_delta,
                        steps=steps,
                    )
                )
            except Exception as err:
                logging.fatal("RDP accounting failed! Double check privacy parameters.")
                if not lenient:
                    raise err

        if accounting_mode in ('all','glw'):
            try:
                manager = accounting_manager.GLWManager(eps_error=self.eps_error)
                privacy_results.update(
                    manager.compute_epsilon(
                        sigma=self.noise_multiplier,
                        sample_rate=self.sample_rate,
                        target_delta=self.target_delta,
                        steps=steps
                    )
                )
            except Exception as err:
                logging.fatal(
                    "Numerical composition of tradeoff functions failed! Double check privacy parameters."
                )
                if not lenient:
                    raise err

        return privacy_results
