"""
This module is a collection of grad samplers 
- methods to calculate per sample grad norms or gradients
for a layer given 1) inputs, AND/OR 2) grad_outputs.

Supports BK (book-keeping) introduced in 
Bu et al. (arXiv:2210.00038)
Differentially Private Optimization on Large Model at Small Cost

and BiTFiT (bias-term fine-tuning) introduced in
Bu et al. (aarXiv:2210.00036)
Differentially Private Bias-Term only Fine-tuning of Foundation Models

Highlights: this code uses the important "mixed ghost norm" trick to achieve its high efficiency,
adapted and improved from 'Scalable and Efficient Training of Large Convolutional Neural Networks with Differential Privacy'
by Bu et al. See their Section 4.

A large portion of this code is adapted Opacus v0.15 (https://github.com/pytorch/opacus), 
from Private-transformers v0.2.3 (https://github.com/lxuechen/private-transformers),
and from Private-vision v0.1.0 (https://github.com/woodyx218/private_vision)
"""

import torch
import transformers.pytorch_utils
from torch import nn
from torch.functional import F
from transformers.models.t5.modeling_t5 import T5LayerNorm


def mixed_ghost_norm(layer,A,B,conv=False):
    # for linear layers, A is activation, B is backprops;
    # for conv layers, A is unfolded activation, B is inverse folded (flattened) backprops;
    if not hasattr(layer, "use_gc"): # use ghost clipping or not
        if conv==False:
            T = torch.prod(torch.Tensor(list(A.shape[1:-1]))).item()
            #assert T == torch.prod(torch.Tensor(list(B.shape[1:-1]))).item()
            d = A.shape[-1]
            p = B.shape[-1]
        else:
            T = A.shape[-1]
            #assert T == B.shape[-1]
            d = A.shape[1]
            p = B.shape[1]
        d_times_p = torch.prod(torch.Tensor(list(layer.weight.size())))
        layer.use_gc = bool(2*T**2 <= d_times_p)
        #assert d*p == d_times_p
        #print(layer,'\n use ghost clip: ',layer.use_gc,'\n T= ',T,';d= ',d,';p= ',p,';2T^2= ',2*T**2,';pd= ',p*d)

def sum_over_all_but_batch_and_last_n(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)


def _light_linear_weight_norm_sample(A, B) -> torch.Tensor:
    """Compute gradient sample norm for the weight matrix in a linear layer."""
    if A.dim() == 2 and B.dim() == 2:
        return _light_linear_weight_norm_sample_non_sequential(A, B)
    elif A.dim() == 3 and B.dim() == 3:
        return _light_linear_weight_norm_sample_sequential(A, B)
    else:
        raise ValueError(f"Unexpected input shape: {A.size()}, grad_output shape: {B.size()}")


@torch.jit.script
def _light_linear_weight_norm_sample_sequential(A, B):
    """Lightweight norm computation in ghost clipping.

    Linear algebra identity trick -- Eq. 3 in the paper.
    """
    #return torch.sqrt((torch.einsum('bTd,bSd->bTS',A,A)*torch.einsum('bTp,bSp->bTS',B,B)).sum(dim=(1, 2)))
    return torch.sqrt((torch.bmm(A, A.transpose(-1, -2)) * torch.bmm(B, B.transpose(-1, -2))).sum(dim=(1, 2)))


@torch.jit.script
def _light_linear_weight_norm_sample_non_sequential(A, B):
    """The Goodfellow trick, i.e., Frobenius norm equal to product of 2-norms."""
    return A.norm(2, dim=1) * B.norm(2, dim=1)

@torch.jit.script
def _light_linear_bias_norm_sample(B):
    if B.dim() == 2:
        return B.norm(2, dim=1)
    elif B.dim() == 3:
        return B.sum(dim=1).norm(2, dim=1)
    else:
        raise ValueError(f"Unexpected grad_output shape: {B.size()}")

def _compute_linear_grad_sample(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, clipping_mode: str) -> None:
    """Computes per sample gradients for `nn.Linear` layer.
    A is activations or layer's input, see autograd_grad_sample line 229; B is output gradient
    This function is written in an unusually bespoke way to avoid using `torch.einsum`.
    """
    if A!=None:
        if clipping_mode in ['MixGhostClip','MixOpt']:
            mixed_ghost_norm(layer, A, B)
        else:
            layer.use_gc=True
        
        
        if A.dim()>3:
            A=torch.flatten(A,start_dim=1,end_dim=-2)
            B=torch.flatten(B,start_dim=1,end_dim=-2)
            
        if layer.use_gc==True:
            #--- compute weight gradient norm
            layer.weight.norm_sample = _light_linear_weight_norm_sample(A, B)
        else:
            ## Or use Line 105 (v0.1.0) https://github.com/lxuechen/private-transformers/blob/main/private_transformers/privacy_utils/supported_layers_grad_samplers.py
            layer.weight.grad_sample = torch.einsum('b...d, b...p-> bpd', A, B).detach()
            layer.weight.norm_sample = torch.sqrt(torch.sum(layer.weight.grad_sample**2, dim=(1, 2)))
            if clipping_mode!='MixOpt':
                del layer.weight.grad_sample
    
    #--- compute bias gradient norm
    if layer.bias is not None:
        layer.bias.norm_sample = _light_linear_bias_norm_sample(B)
        if B.dim() == 3:
            grad_bias = B.sum(dim=1)
        elif B.dim() == 2:
            grad_bias = B
        layer.bias.grad_sample = grad_bias.detach()     


def _compute_Conv1D_grad_sample(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, clipping_mode: str) -> None:
    """Computes per sample gradients for `nn.Linear` layer.
    A is activations or layer's input, see autograd_grad_sample line 229; B is output gradient
    This function is written in an unusually bespoke way to avoid using `torch.einsum`.
    """
    if A!=None:
        if clipping_mode in ['MixGhostClip','MixOpt']:
            mixed_ghost_norm(layer, A, B)
        else:
            layer.use_gc=True
        
        
        if A.dim()>3:
            A=torch.flatten(A,start_dim=1,end_dim=-2)
            B=torch.flatten(B,start_dim=1,end_dim=-2)
            
        if layer.use_gc==True:
            #--- compute weight gradient norm
            layer.weight.norm_sample = _light_linear_weight_norm_sample(A, B)
        else:
            ## Or use Line 105 (v0.1.0) https://github.com/lxuechen/private-transformers/blob/main/private_transformers/privacy_utils/supported_layers_grad_samplers.py
            layer.weight.grad_sample = torch.einsum('b...d, b...p-> bdp', A, B).detach()
            layer.weight.norm_sample = torch.sqrt(torch.sum(layer.weight.grad_sample**2, dim=(1, 2)))
            if clipping_mode!='MixOpt':
                del layer.weight.grad_sample
    
    #--- compute bias gradient norm
    if layer.bias is not None:
        layer.bias.norm_sample = _light_linear_bias_norm_sample(B)
        if B.dim() == 3:
            grad_bias = B.sum(dim=1)
        elif B.dim() == 2:
            grad_bias = B
        layer.bias.grad_sample = grad_bias.detach()   
        
def _compute_layer_norm_grad_sample(
    layer: nn.LayerNorm,
    A: torch.Tensor, B: torch.Tensor, 
    clipping_mode: str) -> None:
    """Computes per sample gradients for normalization layers."""
    if A!=None:
        #--- weight, compute gradient norm
        grad_sample = sum_over_all_but_batch_and_last_n(
            F.layer_norm(A, layer.normalized_shape, eps=layer.eps) * B,
            layer.weight.dim(),
        )
        norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        layer.weight.norm_sample = norm_sample
        layer.weight.grad_sample = grad_sample.detach()
    
    #--- bias, compute gradient norm
    if layer.bias is not None:
        grad_sample = sum_over_all_but_batch_and_last_n(B, layer.bias.dim())        
        layer.bias.norm_sample = grad_sample.flatten(start_dim=1).norm(2, dim=1)
        layer.bias.grad_sample = grad_sample.detach()

      
def _compute_group_norm_grad_sample(
    layer: nn.GroupNorm,
    A: torch.Tensor, B: torch.Tensor, 
    clipping_mode: str
) -> None:
    
    """Computes per sample gradients for normalization layers."""
    if A!=None:
        grad_sample = torch.einsum('ni...->ni',F.group_norm(A, layer.num_groups, eps=layer.eps) * B)
    
        layer.weight.norm_sample = grad_sample.norm(2, dim=1)
        layer.weight.grad_sample = grad_sample.detach()

    if layer.bias is not None:
        grad_sample = torch.einsum('ni...->ni', B)
        layer.bias.norm_sample = grad_sample.norm(2, dim=1)
        layer.bias.grad_sample = grad_sample.detach()

def _compute_instance_norm_grad_sample(
    layer: nn.InstanceNorm2d,
    A: torch.Tensor, B: torch.Tensor, 
    clipping_mode: str
) -> None:
    
    """Computes per sample gradients for normalization layers."""
    if A!=None:
        grad_sample = torch.einsum('ni...->ni',F.instance_norm(A, eps=layer.eps) * B)
    
        layer.weight.norm_sample = grad_sample.norm(2, dim=1)
        layer.weight.grad_sample = grad_sample.detach()

    if layer.bias is not None:
        grad_sample = torch.einsum('ni...->ni', B)
        layer.bias.norm_sample = grad_sample.norm(2, dim=1)
        layer.bias.grad_sample = grad_sample.detach()

def _compute_embedding_grad_sample(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, clipping_mode: str) -> None:
    """Computes per sample gradients for `nn.Embedding` layer."""

    #--- compute gradient norm
    not_AAt: torch.Tensor = ~A[:, :, None].eq(A[:, None, :])
    # Clear the contribution to the norm of the gradient for the padding token.
    #   In vanilla backpropagation, this particular embedding doesn't contribute to the gradient anyway.
    #   For more see 1.10.0 doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    #       'the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.'
    padding_idx = layer.padding_idx
    if padding_idx is not None:
        # The right way to think about the next line of code is that A_i[t, padding_idx] = 0 for all t in [T].
        #   So the entry gets cleared whenever one of A, A^t takes the padding idx.
        not_AAt.bitwise_or_((A[:, :, None] == padding_idx) | (A[:, None, :] == padding_idx))
    norm_sample = torch.sqrt((torch.bmm(B, B.transpose(-1, -2)).masked_fill(not_AAt, 0)).sum(dim=(1, 2)))
    layer.weight.norm_sample = norm_sample


def _compute_conv_grad_sample(layer, A: torch.Tensor, B: torch.Tensor, clipping_mode: str):
    B = B.flatten(2)                                  # F^{-1}(dL/ds)
    # check also https://github.com/woodyx218/private_vision/blob/main/private_vision/privacy_utils/supported_layers_grad_samplers.py
    if A!=None:
        if layer.__class__.__name__=='Conv1d':
            padding = layer.padding if isinstance(
                    layer.padding, tuple) else (*layer.padding, *layer.padding)
            # padded_A = F.pad(A, padding)
            A = F.unfold(A.unsqueeze(-2), kernel_size=(1, *layer.kernel_size),
                                padding=(0, *padding),
                                dilation=(1, *layer.dilation),
                                stride=(1, *layer.stride))
        elif layer.__class__.__name__=='Conv2d':
            A = F.unfold(A, kernel_size=layer.kernel_size,
                                    dilation=layer.dilation, padding=layer.padding,
                                    stride=layer.stride) # U(a)  
        elif layer.__class__.__name__=='Conv3d':
            from opacus.utils import tensor_utils
            A = tensor_utils.unfold3d(A, kernel_size=layer.kernel_size,
                                             dilation=layer.dilation, padding=layer.padding,
                                             stride=layer.stride)
    
        if clipping_mode in ['MixGhostClip','MixOpt']:
            mixed_ghost_norm(layer, A, B,conv=True)
        else:
            layer.use_gc=True
        
        if layer.use_gc==True:
            #--- compute weight gradient norm
            aTa = torch.einsum('bji, bjk -> bik', A, A)
            gTg = torch.einsum('bji, bjk -> bik', B, B)
            #norm_sample = torch.sqrt(torch.einsum('bij, bij -> b', aTa, gTg))
            norm_sample = torch.sqrt((aTa*gTg).sum(dim=(1,2)))    
            layer.weight.norm_sample = norm_sample
        else:
            ## Or use Line 105 https://github.com/lxuechen/private-transformers/blob/main/private_transformers/privacy_utils/supported_layers_grad_samplers.py
            layer.weight.grad_sample = torch.einsum('bd..., bp...-> bpd', A, B).detach()
            layer.weight.norm_sample = torch.sqrt((layer.weight.grad_sample**2).sum(dim=(1, 2)))
            if clipping_mode !='MixOpt':
                del layer.weight.grad_sample

    #--- bias, compute gradient norm
    if layer.bias is not None:
        grad_sample = B.sum(dim=2).detach()
        layer.bias.norm_sample = grad_sample.norm(2, dim=1)
        layer.bias.grad_sample = grad_sample

def _compute_t5_layer_norm_grad_sample(layer: T5LayerNorm, A: torch.Tensor, B: torch.Tensor, clipping_mode: str):
    # `transformers.models.t5.modeling_t5.T5LayerNorm` has single input and output. Unpack singleton tuples.
    # https://github.com/huggingface/transformers/blob/ccc089780415445768bcfd3ac4418cec20353484/src/transformers/models/t5/modeling_t5.py#L248

    assert A.dim() == 3 and B.dim() == 3, (
        "Internal error: T5LayerNorm receiving 2-D tensors, but expected 3-D tensors (sequential inputs)."
    )

    grad_sample = (A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + layer.variance_epsilon) * B).sum(dim=1)
    layer.weight.norm_sample = grad_sample.norm(2, dim=1)

#% compute clipped weight gradient    
def _clip_linear_grad(layer: nn.Linear, A: torch.Tensor, B: torch.Tensor, C) -> None:
    try:
        grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
        del layer.weight.grad_sample
    except:
        grad_weight = torch.einsum('b...d,b...p->pd',A,B)
    return grad_weight

def _clip_normalization_grad(layer, A: torch.Tensor, B: torch.Tensor, C) -> None:
    grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
    del layer.weight.grad_sample
    return grad_weight
        
def _clip_embedding_grad(layer: nn.Embedding, A: torch.Tensor, B: torch.Tensor, C) -> None:
    A = F.one_hot(A, num_classes=layer.weight.shape[0]).to(B)  # (batch_size, seq_len, vocab_dim,)
    grad_weight = torch.einsum('b...d,b...p->dp',A,B)
    ## `torch.nn.Embedding` layers don't accumulate gradient on the padding_idx position.
    ##   We do the same for `grad_sample`.
    if layer.padding_idx is not None:
        # `grad_sample` has size (batch_size, num_vocab, embedding_dim).
        grad_weight[layer.padding_idx, :] = 0.
    return grad_weight
                  
def _clip_Conv1D_grad(layer: transformers.pytorch_utils.Conv1D, A: torch.Tensor, B: torch.Tensor, C) -> None:
    try:
        grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
        del layer.weight.grad_sample
    except:
        grad_weight = torch.einsum('b...d,b...p->dp',A,B)
    return grad_weight

def _clip_conv_grad(layer, A: torch.Tensor, B: torch.Tensor, C):
    B = B.flatten(2)                                  # F^{-1}(dL/ds)
    try:
        grad_weight = torch.einsum('b...,b->...',layer.weight.grad_sample,C)
        del layer.weight.grad_sample
    except:
        if type(layer)==nn.Conv1d:
            padding = layer.padding if isinstance(
                    layer.padding, tuple) else (*layer.padding, *layer.padding)
            # padded_A = F.pad(A, padding)
            A = F.unfold(A.unsqueeze(-2), kernel_size=(1, *layer.kernel_size),
                                padding=(0, *padding),
                                dilation=(1, *layer.dilation),
                                stride=(1, *layer.stride))
        elif type(layer)==nn.Conv2d:
            A = F.unfold(A, kernel_size=layer.kernel_size,
                                    dilation=layer.dilation, padding=layer.padding,
                                    stride=layer.stride) # U(a)
        elif type(layer)==nn.Conv3d:
            from opacus.utils import tensor_utils
            A = tensor_utils.unfold3d(A, kernel_size=layer.kernel_size,
                                             dilation=layer.dilation, padding=layer.padding,
                                             stride=layer.stride)
        
        grad_weight = torch.einsum('bDT,bpT->pD',A,B)
        #grad_weight = torch.bmm(B, A.permute(0, 2, 1)).sum(dim=0)      

    grad_weight=grad_weight.view(-1, *layer.weight.shape)[0]
    return grad_weight

def _clip_t5_layer_norm_grad(layer: T5LayerNorm, A: torch.Tensor, B: torch.Tensor, clipping_mode: str):
    grad_weight = (A * torch.rsqrt(A.pow(2).mean(-1, keepdim=True) + layer.variance_epsilon) * B).sum(dim=1)
    return grad_weight


_supported_layers_norm_sample_AND_clipping = {
    nn.Embedding: (_compute_embedding_grad_sample, _clip_embedding_grad),
    nn.Linear: (_compute_linear_grad_sample, _clip_linear_grad),
    nn.Conv1d: (_compute_conv_grad_sample, _clip_conv_grad),
    nn.Conv2d: (_compute_conv_grad_sample, _clip_conv_grad),
    nn.LayerNorm: (_compute_layer_norm_grad_sample, _clip_normalization_grad),
    nn.GroupNorm: (_compute_group_norm_grad_sample, _clip_normalization_grad),
    nn.InstanceNorm1d: (_compute_instance_norm_grad_sample, _clip_normalization_grad),
    nn.InstanceNorm2d: (_compute_instance_norm_grad_sample, _clip_normalization_grad),
    nn.InstanceNorm3d: (_compute_instance_norm_grad_sample, _clip_normalization_grad),
    transformers.pytorch_utils.Conv1D: (_compute_Conv1D_grad_sample, _clip_Conv1D_grad),# Conv1D's weight is transposed to nn.Linear's, but this does not matter for the norm
    transformers.models.t5.modeling_t5.T5LayerNorm: (_compute_t5_layer_norm_grad_sample, _clip_t5_layer_norm_grad),
}

#%  we need param.summed_clipped_grad to avoid contamination from non-private .grad
def _create_or_extend_summed_clipped_grad(param: torch.Tensor, summed_clipped_grad: torch.Tensor) -> None:
    """Adds summed clipped gradient (not per-sample) to param.summed_clipped_grad or accumulate the existing tensor."""
    
    assert summed_clipped_grad.shape == param.shape, f"summed clipped grad.size()={summed_clipped_grad.size()}, param.size()={param.size()}"

    if hasattr(param, "summed_clipped_grad"):
        param.summed_clipped_grad += summed_clipped_grad.detach()
    else:
        param.summed_clipped_grad = summed_clipped_grad.detach();#print(torch.normal(0,1,size=(1,1)))
        #print(param.summed_clipped_grad.dtype)

#%  we need param.private_grad stores either noise+first micro-batch summed_clipped_grad or only summed_clipped_grad
def _create_or_extend_private_grad(param: torch.Tensor, summed_clipped_grad: torch.Tensor) -> None:
    """Adds summed clipped gradient (not per-sample) to param.summed_clipped_grad or accumulate the existing tensor."""

    assert summed_clipped_grad.shape == param.shape, f"summed clipped grad.size()={summed_clipped_grad.size()}, param.size()={param.size()}"
    if hasattr(param, "private_grad"):
        param.private_grad = summed_clipped_grad.detach()
    else:
        param.private_grad = summed_clipped_grad.detach()+torch.normal(mean=0, std=param.noise,size=param.size(), device=param.device, dtype=param.dtype)
