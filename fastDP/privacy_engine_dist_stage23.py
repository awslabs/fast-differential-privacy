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

from . import autograd_grad_sample_dist, transformers_support
from .accounting import accounting_manager
from torch.functional import F
import transformers
from .supported_layers_grad_samplers import _supported_layers_norm_sample_AND_clipping


class PrivacyEngine_Distributed_Stage_2_and_3(object):
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
        noise_multiplier: Optional[float] = None,
        target_epsilon: Optional[float] = None,
        target_delta: Optional[float] = None,
        alphas: Sequence[float] = accounting_manager.DEFAULT_ALPHAS,
        named_params: Optional[Sequence] = None,
        numerical_stability_constant=None,
        accounting_mode="rdp",
        eps_error=0.05,
        clipping_mode='MixOpt',
        clipping_fn='automatic',
        loss_reduction='mean',
        clipping_style='layer-wise',
        num_GPUs=1,
        torch_seed_is_fixed=False,
        **unused_kwargs,
    ):

        """Initialize the engine.

        Args:
            module: The PyTorch module for which per-sample gradient is required.
                Setting the `requires_grad` attribute of a parameter to False
                disables the per-sample gradient accumulation.
            batch_size: The expected size of Poisson-sampled batch, i.e., the lot size.
            sample_size: Size of dataset.
            max_grad_norm: The maximum 2-norm for gradient clipping.
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
            loss_reduction: Reduction of loss, one of 'sum' and 'mean'.
            clipping_style: The clipping style to use. One of 'all-layer', 'layer-wise', 'param-wise' or an un-ordered list of layer names that represent blocks' head layer
        """
        del unused_kwargs
        super(PrivacyEngine_Distributed_Stage_2_and_3, self).__init__()

        if clipping_mode not in ['ghost','MixGhostClip','MixOpt']:
            raise ValueError(f"Unknown clipping mode {clipping_mode}. Expected one of 'ghost','MixGhostClip','MixOpt'.")
        if accounting_mode not in ("rdp", "all",'glw'):
            raise ValueError(f"Unknown accounting mode: {accounting_mode}. Expected one of 'rdp', 'all','glw'.")
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

        # Internals.
        self.steps = 0  # Tracks privacy spending.
        
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

        #-----
        def _supported_and_trainable(layer):            
            if type(layer) in _supported_layers_norm_sample_AND_clipping and ((hasattr(layer,'weight') and hasattr(layer.weight,'requires_grad') and layer.weight.requires_grad) or (hasattr(layer,'bias') and hasattr(layer.bias,'requires_grad') and layer.bias.requires_grad)):
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
            self.n_components+=sum([1 for p in layer.parameters() if p.requires_grad])
        print("Number of trainable components: ",self.n_components, "; Number of trainable layers: ",self.n_layers)


        #-----
        print('>>>>>>>>>>>>>>>>> Applying ',clipping_fn, ' per-sample gradient clipping.')
        self.clipping_fn = clipping_fn
        if numerical_stability_constant!=None:
            self.numerical_stability_constant = numerical_stability_constant
        elif self.clipping_fn=='automatic':
            self.max_grad_norm = 1.0 # max_grad_norm does not matterin automatic clipping; this is necessary for step()
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

        # create list of block head layers        
        if isinstance(clipping_style,list):
            self.clipping_style='group-wise'
            self.block_heads=clipping_style
        else:            
            self.clipping_style=clipping_style
            self.block_heads=[]
        
            if self.clipping_style=='all-layer':
                self.block_heads.append(self.named_layers[0][0])
            elif self.clipping_style in ['layer-wise','param-wise']:
                self.block_heads = [name for (name,layer) in self.named_layers]
        #print(">>>>>>>>>>>>>>>>> Block heads for per-sample gradient clipping are defined as:", self.block_heads)

        transformers_support.forward_swapper(module=module)  # fix the position embeddings broadcast issue.
        
        autograd_grad_sample_dist.add_hooks(model=self.module, loss_reduction=self.loss_reduction, 
                                       clipping_mode=self.clipping_mode, bias_only=self.bias_only,
                                       clipping_style=self.clipping_style, block_heads=self.block_heads,
                                       named_params=self.named_params, named_layers=self.named_layers,
                                       clipping_fn=self.clipping_fn, 
                                       numerical_stability_constant=self.numerical_stability_constant,
                                       max_grad_norm_layerwise=self.max_grad_norm_layerwise)


        def get_privacy_spent(_self, **kwargs):
            return _self.privacy_engine.get_privacy_spent(**kwargs)

        # Make getting info easier.
        self.module.get_privacy_spent = types.MethodType(get_privacy_spent, self.module)

        self.module.privacy_engine = self


        # deepspeed stage 3 modification-----------
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3, print_rank_0

        def zero_grad_DP_stage3(self, set_grads_to_None=True, set_to_none=None):
            """
            Zero FP16 parameter grads.
            """
            if set_to_none is not None:
                # In transformers 4.29, set_grads_to_None is renamed to set_to_none
                set_grads_to_None = set_to_none
            #print(self.micro_step_id)
            self.micro_step_id = 0

            # FP32 grad should never exist.
            # For speed, set model fp16 grad to None by default
            for group in self.fp16_groups:
                for p in group:
                    if set_grads_to_None:
                        if p.grad is not None and p.grad.is_cuda:
                            p.grad.record_stream(torch.cuda.current_stream())
                        p.grad = None
                    else:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    if hasattr(p,'private_grad'): #zhiqi: del private_grad so next step adds noise in supported...py
                        del p.private_grad
                        
        DeepSpeedZeroOptimizer_Stage3.zero_grad = zero_grad_DP_stage3
        
        def create_reduce_and_remove_grad_hooks_DP_stage3(self):
            print_rank_0(f'[Begin] Create gradient reduction hooks')
            self.grad_accs = []
            for i, param_group in enumerate(self.fp16_groups):
                for param in param_group:
                    if param.requires_grad:
                        #print_rank_0(f" Before all gather {param.device}, {param.shape}")
    
                        # The hook must be created in un-partitioned parameter
                        param.all_gather()
    
                        #print(f"After all gather {param.device}, {param.shape}")
                        def wrapper(param, i):
                            param_tmp = param.expand_as(param)
                            grad_acc = param_tmp.grad_fn.next_functions[0][0]
    
                            def reduce_partition_and_remove_grads(*notneeded):
                                #!!!!!!!!
                                if hasattr(param,'private_grad'):
                                    param.grad=torch.nan_to_num(param.private_grad).contiguous() / param.batch_size * self.loss_scale # it works
                                    param.private_grad = None # release memory
                                else:
                                    param.grad.zero_()
                                #!!!!!!!!

                                self.reduce_ready_partitions_and_remove_grads(param, i)
    
                            grad_acc.register_hook(reduce_partition_and_remove_grads)
                            self.grad_accs.append(grad_acc)
    
                        #print(f"param grad fn {param.expand_as(param).grad_fn}")
                        wrapper(param, i)
    
                        # Partition the parameter after creating the hook
                        param.partition()
            print_rank_0(f'[End] Create gradient reduction hooks')

        DeepSpeedZeroOptimizer_Stage3.create_reduce_and_remove_grad_hooks = create_reduce_and_remove_grad_hooks_DP_stage3
        
        
        # deepspeed stage 2 modification-----------
        from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

        def zero_grad_DP_stage2(self, set_grads_to_None=True, set_to_none=None):
            """
            Zero FP16 parameter grads.
            """
            if set_to_none is not None:
                # In transformers 4.29, set_grads_to_None is renamed to set_to_none
                set_grads_to_None = set_to_none
            #print(self.micro_step_id)

            # FP32 grad should never exist.
            # For speed, set model fp16 grad to None by default
            for group in self.bit16_groups:
                for p in group:
                    if set_grads_to_None:
                        p.grad = None  # epilogue and in step
                    else:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()
                    if hasattr(p,'private_grad') and self.micro_step_id == -1:#zhiqi, https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py#L1752
                        del p.private_grad

        DeepSpeedZeroOptimizer.zero_grad = zero_grad_DP_stage2

        def create_reduce_and_remove_grad_hooks_DP_stage2(self):
            self.grad_accs = []
            for i, param_group in enumerate(self.bit16_groups):
                for param in param_group:
                    if param.requires_grad:

                        def wrapper(param, i):
                            param_tmp = param.expand_as(param)
                            grad_acc = param_tmp.grad_fn.next_functions[0][0]

                            def reduce_partition_and_remove_grads(*notneeded):
                                #!!!!!!!!
                                if hasattr(param,'private_grad'):
                                    param.grad=torch.nan_to_num(param.private_grad).contiguous() / param.batch_size * self.loss_scale # it works
                                    param.private_grad = None # release memory
                                else:
                                    param.grad.zero_()
                                #!!!!!!!!

                                self.reduce_ready_partitions_and_remove_grads(param, i)

                            grad_acc.register_hook(reduce_partition_and_remove_grads)
                            self.grad_accs.append(grad_acc)

                        wrapper(param, i)

        DeepSpeedZeroOptimizer.create_reduce_and_remove_grad_hooks = create_reduce_and_remove_grad_hooks_DP_stage2

        # FSDP modification----------- fairscale == 0.4.13
        from fairscale.nn.data_parallel import FullyShardedDataParallel
        from fairscale.nn.data_parallel import TrainingState
        import functools
        from typing import Any
        from torch.nn.parameter import Parameter
        from fairscale.internal.parallel import chunk_and_pad
        
        @torch.no_grad()
        def _post_backward_hook(self, param: Parameter, *unused: Any) -> None:
            """
            At the start of :func:`_post_backward_hook`, ``param.grad`` contains the
            full gradient for the local batch. The reduce-scatter op will replace
            ``param.grad`` with a single shard of the summed gradient across all
            GPUs. This shard will align with the current GPU rank. For example::
                before reduce_scatter:
                    param.grad (GPU #0): [1, 2, 3, 4]
                    param.grad (GPU #1): [5, 6, 7, 8]
                after reduce_scatter:
                    param.grad (GPU #0): [6, 8]    # 1+5, 2+6
                    param.grad (GPU #1): [10, 12]  # 3+7, 4+8
            The local GPU's ``optim.step`` is responsible for updating a single
            shard of params, also corresponding to the current GPU's rank. This
            alignment is created by :func:`_shard_parameters_`, which ensures that
            the local optimizer only sees the relevant parameter shard.
            """
            # First hook callback will see PRE state. If we have multiple params,
            # then subsequent hook callbacks will see POST state.
            self.assert_state([TrainingState.BACKWARD_PRE, TrainingState.BACKWARD_POST])
            self.training_state = TrainingState.BACKWARD_POST
            
            #--------------- zhiqi modification
            if hasattr(param,'private_grad') and param.grad!=None:
                #print(hasattr(param,'noise'),hasattr(param,'private_grad'),param==None,param.grad==None,param.private_grad==None)
                #print(param.shape,param.grad.shape,param.private_grad.shape)
                param.grad=torch.nan_to_num(param.private_grad) / param.batch_size # it works
                param.private_grad = None # release memory
            elif param.grad != None:
                param.grad.zero_()
            #--------------------
            if hasattr(param, "_linked_param"):
                # This links to a shared param. We should try to finalize the linked param here.
                # This is done by module code to ensure correct gradient computation.
                # p._is_shared and p._linked_param are closely related but not the same.
                # See fairscale/experimental/nn/mevo.py.
                assert param.shape == (1,), param.shape  # This param should have this special dim.
                # If the _is_shared flag is set, then this shared weight is indeed being
                # shared between different FSDP wrappers. Otherwise, they are linked but
                # likely in the same FSDP wrapper, which means we shouldn't finalize the
                # linked param..
                if hasattr(param._linked_param, "_is_shared") and param._linked_param._is_shared:
                    # param._linked_param may or may not have .grad since this callback
                    # could happen multiple times to support #918. Since we check `if param.grad is None`
                    # below anyway, this is OK.
                    param = param._linked_param
    
            if param.grad is None:
                return
    
            if param.grad.requires_grad:
                raise RuntimeError("FSDP only works with gradients that don't require gradients")
    
            if self._require_backward_grad_sync or self.reshard_after_forward:
                # Free full params. As a special case, we don't free the full params
                # when in a ``no_sync`` context (as inversely indicated by
                # ``self._require_backward_grad_sync``), since the params will not
                # get updated before the next forward. This saves networking
                # bandwidth but uses more GPU memory.
                self._free_full_params([param])
    
            if self.mixed_precision:
                # This is a no-op if reshard_after_forward is True, since we already
                # free the param shard when rebuilding the full params in the
                # pre_backward_hook.
                self._free_fp16_param_shard([param])
    
            # Switch to FP32 shard after backward.
            self._use_fp32_param_shard([param])
    
            if not self._require_backward_grad_sync:
                return
    
            # Wait for all work in the current stream to finish, then start the
            # reductions in post_backward stream.
            self._streams["post_backward"].wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._streams["post_backward"]):
                orig_grad_data = param.grad.data
    
                if self.mixed_precision and self.fp32_reduce_scatter:
                    # Cast grad to FP32.
                    param.grad.data = param.grad.data.to(param.dtype)
    
                if self.gradient_predivide_factor > 1:
                    # Average grad by world_size for consistency with PyTorch DDP.
                    param.grad.data.div_(self.gradient_predivide_factor)
    
                if param._is_sharded:
                    assert self._reducer is not None
                    # Save the unsharded grad for reduction. We will asynchronously accumulate the reduced gradient into
                    # param._saved_grad_shard. If this FSDP module was called multiple times it's possible that multiple
                    # gradient reductions will happen in an undefined order. But addition commutes, so this order doesn't
                    # matter, neglecting rounding.
                    grad = param.grad.data
                    # Clear grad on the tensor, so any repeated gradient computations do not interfere with this reduction.
                    #
                    # The effect on memory consumption is not usually significant. No extra memory is allocated if this
                    # module is called only once, reduction happens quickly, or the tensor is bucketed. If the module is
                    # called multiple times, and the backwards pass runs far enough ahead of the `post_backward` stream,
                    # then we can end up with multiple unsharded gradients allocated and queued for reduction.
                    #
                    # We could guard against this by using CUDA events (see record_event, wait_event in torch.cuda.Stream).
                    # This ensures the `default` stream will wait for the `post_backward` stream to complete the last
                    # reduction for this module, before scheduling additional reduction work. Then at most there are two
                    # unsharded gradients allocated; one for a pending reduction, and one for gradient computation.
                    param.grad = None
                    callback_fn = functools.partial(self._post_reduction_hook, param)
                    grad_chunks = chunk_and_pad(grad, self.process_group_reduce_scatter.size())
                    self._reducer.reduce_scatter_async(
                        grad_chunks, group=self.process_group_reduce_scatter, callback_fn=callback_fn
                    )
                else:
                    # Currently the only way for _is_sharded to be False is if
                    # world_size == 1. This could be relaxed in the future, in which
                    # case grads should be all-reduced here.
                    assert self.world_size == 1
                    self._post_reduction_hook(param, param.grad.data)
    
                # After _post_backward_hook returns, orig_grad_data will eventually
                # go out of scope, at which point it could otherwise be freed for
                # further reuse by the main stream while the div/reduce_scatter/copy
                # are underway in the post_backward stream. See:
                # github.com/NVIDIA/apex/blob/master/apex/parallel/distributed.py
                orig_grad_data.record_stream(self._streams["post_backward"])
    
        FullyShardedDataParallel._post_backward_hook = _post_backward_hook
    
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
