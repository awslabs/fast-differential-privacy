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

from . import autograd_grad_sample, transformers_support
from .accounting import accounting_manager
from torch.functional import F
import transformers
from .supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping


class PrivacyEngine(object):
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
        max_grad_norm: float = 1.,
        epochs: Optional[Union[int, float]] = None,
        num_steps: Optional[Union[int, float]] = None,
        noise_multiplier: Optional[float] = None,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
        alphas: Sequence[float] = accounting_manager.DEFAULT_ALPHAS,
        record_snr: bool = False,
        named_params: Optional[Sequence] = None,
        numerical_stability_constant=None,
        accounting_mode="rdp",
        eps_error=0.05,
        clipping_mode='MixOpt',
        clipping_fn='automatic',
        loss_reduction='mean',
        origin_params=None,
        clipping_style='all-layer',
        num_GPUs=1,
        torch_seed_is_fixed=False,
        **unused_kwargs,
    ):

        """Initialize the engine.

        Args:
            module: The PyTorch module for which per-sample gradient is required.
                Setting the `requires_grad` attribute of a parameter to False
                disables the per-sample gradient accumulation.
            batch_size: The expected size of a logical batch.
            sample_size: Size of dataset.
            max_grad_norm: The maximum 2-norm for gradient clipping.
            epochs: The number of epochs for training.
            num_steps: The number of steps for training, only used if epochs is None.
            noise_multiplier: The extra multiplier for DP-SGD noise.
            target_epsilon: The target privacy spending.
                Only used to estimate the `noise_multiplier` if it is not set.
            target_delta: The target failure probability.
                Defaults to sample_size ** -1.1 if not set.!!!!!!!!!!!!
            alphas: The RDP orders for (ε, δ)-DP conversion. Useless if not accounting in RDP.
            record_snr: Record and report the signal-to-noise ratio --
                ratio between norm of summed clipped gradient and norm of noise vector.
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
            loss_reduction: Reduction of loss, one of 'sum' and 'mean'.
            origin_params: Specifies which are origin parameters as described in ghost differentiation. Can be None or list of parameter names
                ['_embeddings','wte','wpe'] is used for roberta and GPT2. For general model, can set to first layer's bias or weight.
            clipping_style: The clipping style to use. One of 'all-layer', 'layer-wise', 'param-wise' or an un-ordered list of layer names that represent blocks' head layer
        """
        del unused_kwargs
        super(PrivacyEngine, self).__init__()

        if clipping_mode not in ['ghost','MixGhostClip','MixOpt']:
            raise ValueError(f"Unknown clipping mode {clipping_mode}. Expected one of 'ghost','MixGhostClip','MixOpt'.")
        if accounting_mode not in ("rdp", "all",'glw'):
            raise ValueError(f"Unknown accounting mode: {accounting_mode}. Expected one of 'rdp', 'all','glw'.")
        if epochs is None:
            if num_steps is not None:
                epochs=num_steps/sample_size*batch_size
            else:
                raise ValueError(f"Number of training epochs and training steps are not defined.")
        if epochs <= 0.0 and noise_multiplier is None:
            raise ValueError(f"Number of training epochs cannot be non-positive, but found epochs={epochs}")

        # Privacy parameters.
        sample_rate = batch_size / sample_size
        if target_delta is None:
            target_delta = 1 / (2 * sample_size)
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
        self.max_grad_norm = max_grad_norm

        self.epochs = epochs
        self.noise_multiplier = noise_multiplier
        self.effective_noise_multiplier = noise_multiplier / batch_size
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.alphas = alphas
        self.eps_error = eps_error
        self.accounting_mode = accounting_mode
        self.record_snr = record_snr

        # Internals.
        self.steps = 0  # Tracks privacy spending.

        # Recording.
        self.max_clip = None
        self.min_clip = None
        self.med_clip = None
        self.signal = None
        self.noise = None
        self.snr = None
        self.noise_limit = None
        
        # Record parameters.
        self.module = module
        if named_params is None:
            self.named_params = list(
                (name, param) for (name, param) in module.named_parameters() if param.requires_grad
            )
        else:
            self.named_params = named_params
        self.num_params = sum(param.numel() for _, param in self.named_params)

        self._locked = False  # lock the part where noisy gradients is created (in `self.step`) if True.


        #----- ghost differentiation trick through origin parameter
        for name,param in module.named_parameters():
            param.initially_requires_grad=bool(param.requires_grad)
            if origin_params!=None:
                param.requires_grad=param.initially_requires_grad and any([i in name for i in origin_params]) # only requires grad if it is origin and initially requires grad

        if origin_params!=None:
            print('Using origin parameters for the ghost differentiation trick......')

        #-----
        def _supported_and_trainable(layer):            
            if type(layer) in _supported_layers_norm_sample_AND_clipping and ((hasattr(layer,'weight') and hasattr(layer.weight,'initially_requires_grad') and layer.weight.initially_requires_grad) or (hasattr(layer,'bias') and hasattr(layer.bias,'initially_requires_grad') and layer.bias.initially_requires_grad)):
                return True
            return False

        # store layer's name and create list of named layers for blockwise clipping
        self.named_layers=[]
        for name,layer in module.named_modules():
            if _supported_and_trainable(layer):
                self.named_layers.append((name,layer))

        self.n_layers=len(self.named_layers) #sum(1 for layer in module.modules() if autograd_grad_sample.requires_grad(layer) and hasattr(layer,'weight'))
        
        self.n_components=0
        for name, layer in self.named_layers:
            self.n_components+=sum([1 for p in layer.parameters() if p.initially_requires_grad])
        print("Number of trainable components: ",self.n_components, "; Number of trainable layers: ",self.n_layers)


        #-----
        print('>>>>>>>>>>>>>>>>> Applying ',clipping_fn, ' per-sample gradient clipping.')
        self.clipping_fn = clipping_fn
        if numerical_stability_constant!=None:
            self.numerical_stability_constant = numerical_stability_constant
        elif self.clipping_fn=='automatic':
            self.max_grad_norm = 1. # max_grad_norm does not matterin automatic clipping; this is necessary for step()
            self.numerical_stability_constant=1e-2
        else:
            self.numerical_stability_constant=1e-6
        
        if clipping_style=='layer-wise':
            self.max_grad_norm_layerwise = self.max_grad_norm / math.sqrt(self.n_layers)
        elif clipping_style=='param-wise':
            self.max_grad_norm_layerwise = self.max_grad_norm / math.sqrt(self.n_components)
        elif clipping_style=='all-layer':
            self.max_grad_norm_layerwise=self.max_grad_norm
        else:
            self.max_grad_norm_layerwise=self.max_grad_norm / math.sqrt(len(clipping_style))

        for name,param in module.named_parameters():
            param.batch_size = self.batch_size
            if torch_seed_is_fixed == True:
                param.noise = self.noise_multiplier*self.max_grad_norm / num_GPUs
            else:
                param.noise = self.noise_multiplier*self.max_grad_norm / math.sqrt(num_GPUs)

        self.loss_reduction = loss_reduction
        self.clipping_mode = clipping_mode
        
        #----- determine whether training with BiTFiT
        self.bias_only=True
        for name,param in self.named_params:
            if '.bias' not in name and param.requires_grad:
                self.bias_only=False; break

        if self.bias_only:
            origin_params=None # do not use origin parameters for BiTFiT
            

        
        # create list of block head layers        
        if isinstance(clipping_style,list):
            self.clipping_style='block-wise'
            self.block_heads=clipping_style
        else:            
            self.clipping_style=clipping_style
            self.block_heads=[]
        
            if self.clipping_style=='all-layer':
                self.block_heads.append(self.named_layers[0][0])
            elif self.clipping_style in ['layer-wise','param-wise']:
                self.block_heads = [name for (name,layer) in self.named_layers]
        print(">>>>>>>>>>>>>>>>> Block heads for per-sample gradient clipping are defined as:", self.block_heads)

        transformers_support.forward_swapper(module=module)  # fix the position embeddings broadcast issue.

        autograd_grad_sample.add_hooks(model=self.module, loss_reduction=self.loss_reduction, 
                                       clipping_mode=self.clipping_mode, bias_only=self.bias_only,
                                       clipping_style=self.clipping_style, block_heads=self.block_heads,
                                       named_params=self.named_params, named_layers=self.named_layers,
                                       clipping_fn=self.clipping_fn, 
                                       numerical_stability_constant=self.numerical_stability_constant,
                                       max_grad_norm_layerwise=self.max_grad_norm_layerwise)


        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            return _self.privacy_engine.get_training_stats(**kwargs)

        # Make getting info easier.
        self.module.get_privacy_spent = types.MethodType(get_privacy_spent, self.module)
        self.module.get_training_stats = types.MethodType(get_training_stats, self.module)

        self.module.privacy_engine = self


        # deepspeed stage 1 modification-----------
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
        from deepspeed import comm as dist

        def reduce_gradients_DP_stage_1(self, pipeline_parallel=False):
            world_size = dist.get_world_size(self.dp_process_group)
            my_rank = dist.get_rank(self.dp_process_group)

            # with PP we must create ipg buffer, since backward is handled outside zero
            if pipeline_parallel and self.contiguous_gradients:
                self.ipg_buffer = []
                buf_0 = torch.empty(int(self.reduce_bucket_size),
                                    dtype=self.dtype,
                                    device=torch.cuda.current_device())
                self.ipg_buffer.append(buf_0)
                self.ipg_index = 0

            if not self.overlap_comm:
                for i, group in enumerate(self.bit16_groups):
                    for param in group:
                        if param.grad is not None:
                            if hasattr(param,'summed_clipped_grad'):
                                param.grad = torch.nan_to_num(param.summed_clipped_grad).contiguous()+torch.normal(mean=0, std=param.noise,size=param.size(), device=param.device, dtype=param.dtype);#print('++++++++++++++++++',param.grad.dtype)
                                del param.summed_clipped_grad # release memory
                                param.grad = param.grad / param.batch_size * self.loss_scale # it works
                            else:
                                param.grad.zero_()

                            self.reduce_ready_partitions_and_remove_grads(param, i)
            # reduce any pending grads in either hook/non-hook case
            self.overlapping_partition_gradients_reduce_epilogue()

        DeepSpeedZeroOptimizer.reduce_gradients = reduce_gradients_DP_stage_1

    def lock(self):
        """Run this after noisy clipped gradient is created to prevent tampering with it before parameter update."""
        self._locked = True

    def unlock(self):
        """Run this after parameter update to allow creation of noisy gradient for next step"""
        self._locked = False

    def attach(self, optimizer):

        # Override step.
        def dp_step(_self, **kwargs):
            closure = kwargs.pop("closure", None)
            
            _self.zero_grad()         # make sure no non-private grad remains
            _self.privacy_engine._create_noisy_clipped_gradient(**kwargs)
            _self.original_step(closure=closure)
            _self.privacy_engine.unlock()  # Only enable creating new grads once parameters are updated.
            _self.privacy_engine.steps += 1


        optimizer.privacy_engine = self

        optimizer.original_step = optimizer.step
        optimizer.step = types.MethodType(dp_step, optimizer)        

        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        def get_training_stats(_self, **kwargs):
            return _self.privacy_engine.get_training_stats(**kwargs)

        # Make getting info easier.
        optimizer.get_privacy_spent = types.MethodType(get_privacy_spent, optimizer)
        optimizer.get_training_stats = types.MethodType(get_training_stats, optimizer)


    def detach(self):
        optimizer = self.optimizer
        optimizer.step = optimizer.original_step
        delattr(optimizer, "privacy_engine")
        delattr(optimizer, "original_step")
        delattr(optimizer, "get_privacy_spent")
        delattr(optimizer, "get_training_stats")

        module = self.module
        autograd_grad_sample.remove_hooks(module)
        module.zero_grad()

        for layer in self.module.modules():
            if hasattr(layer,'activations'):
                del layer.activations
            if hasattr(layer,'backprops'):
                del layer.backprops
        for name,param in self.named_params:
            param.requires_grad=bool(param.initially_requires_grad)
            if hasattr(param,'summed_clipped_grad'):
                del param.summed_clipped_grad

                    
    def _create_noisy_clipped_gradient(self):
        """Create noisy clipped gradient for `optimizer.step`."""
        
        unsupported_param_name=[]
        for name,param in list(self.named_params):#https://thispointer.com/python-remove-elements-from-a-list-while-iterating/#1
            if not hasattr(param, 'summed_clipped_grad'):
                unsupported_param_name.append(name)
                self.named_params.remove((name,param)) # very helpful for models that are not 100% supported, e.g. in timm
        if unsupported_param_name!=[]:
            print(unsupported_param_name, 'are not supported by privacy engine; these parameters are not requiring gradient nor updated.')
                
        signals, noises = [], []
        
        for name,param in self.named_params:
            param.grad = param.summed_clipped_grad  # Ultra important to override `.grad`.
            del param.summed_clipped_grad

            if self.record_snr:
                signals.append(param.grad.reshape(-1).norm(2))
            if self.noise_multiplier > 0 and self.max_grad_norm > 0:
                param.grad += torch.normal(
                    mean=0,
                    std=param.noise,
                    size=param.size(),
                    device=param.device,
                    dtype=param.dtype,
                )
            if self.loss_reduction=='mean':
                param.grad /= self.batch_size                

        if self.record_snr and len(noises) > 0:
            self.signal, self.noise = tuple(torch.stack(lst).norm(2).item() for lst in (signals, noises))
            self.noise_limit = math.sqrt(self.num_params) * self.noise_multiplier * self.max_grad_norm
            self.snr = self.signal / self.noise
        else:
            self.snr = math.inf  # Undefined!

        self.lock()  # Make creating new gradients impossible, unless optimizer.step is called.

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

    def get_training_stats(self):
        """Get the clipping, signal, and noise statistics."""
        return {
            "med_clip": self.med_clip,
            "max_clip": self.max_clip,
            "min_clip": self.min_clip,
            "snr": self.snr,
            "signal": self.signal,
            "noise": self.noise,
            "noise_limit": self.noise_limit,
        }

    def __repr__(self):
        return (
            f"PrivacyEngine(\n"
            f"  target_epsilon={self.target_epsilon:.6f}, \n"
            f"  target_delta={self.target_delta:.6f}, \n"
            f"  noise_multiplier={self.noise_multiplier:.6f}, \n"
            f"  effective_noise_multiplier={self.effective_noise_multiplier:.6f}, \n"
            f"  epochs={self.epochs}, \n"
            f"  max_grad_norm={self.max_grad_norm}, \n"
            f"  sample_rate={self.sample_rate}, \n"
            f"  batch_size={self.batch_size}, \n"
            f"  accounting_mode={self.accounting_mode}, \n"
            f"  clipping_mode={self.clipping_mode}\n"
            f")"
        )
