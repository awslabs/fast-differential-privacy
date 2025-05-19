import torch
import torch.nn.functional as F
import math

from torch.autograd import Function

from typing import Optional, List, Tuple, Union
from torch.nn.parameter import Parameter
from torch.nn import Module,init
from torch import Tensor, Size
import transformers
#%% extending nn.functional with per-sample gradient clipping, through https://pytorch.org/docs/stable/notes/extending.html
''' 
STEPS: (1) create fautograd.Function with new forward and backward, 
           copy F.XX (e.g. Linear) function or write your own to write forward, 
           copy fastDP/supported_layers_grad_samplers.py (compute... and clip...) to write backward
           note that computing param gradient from output grad and input is generally challenging
       (2) create nn.Module with new init and forward,
           note that module forward calls Function forward
       (3) create replacing functions according to nn.Module format
'''

# Inherit from Function    
@torch.jit.script
def _linear_weight_norm_sample_sequential_ghostnorm(input, grad_output, grad_norm2):
    """Lightweight norm computation in ghost clipping.

    Linear algebra identity trick -- Eq. 3 in the paper.
    """
    grad_norm2 = grad_norm2 + (torch.bmm(input, input.transpose(-1, -2)) * torch.bmm(grad_output, grad_output.transpose(-1, -2))).sum(dim=(1, 2))
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    grad_output = torch.einsum('B,B...->B...',clip_factor,grad_output)
    return torch.einsum('B...p,B...d->pd',grad_output,input), clip_factor

@torch.jit.script
def _linear_weight_norm_sample_sequential_ghostnorm_T(input, grad_output, grad_norm2):
    """Lightweight norm computation in ghost clipping.

    Linear algebra identity trick -- Eq. 3 in the paper.
    """
    grad_norm2 = grad_norm2 + (torch.bmm(input, input.transpose(-1, -2)) * torch.bmm(grad_output, grad_output.transpose(-1, -2))).sum(dim=(1, 2))
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    grad_output = torch.einsum('B,B...->B...',clip_factor,grad_output)
    return torch.einsum('B...p,B...d->dp',grad_output,input), clip_factor

@torch.jit.script
def _linear_weight_norm_sample_non_sequential_ghostnorm(input, grad_output, grad_norm2):
    """The Goodfellow trick, i.e., Frobenius norm equal to product of 2-norms."""
    grad_norm2 = grad_norm2 + (input.norm(2, dim=1) * grad_output.norm(2, dim=1))**2
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    grad_output = torch.einsum('B,B...->B...',clip_factor,grad_output)
    return torch.einsum('B...p,B...d->pd',grad_output,input), clip_factor

@torch.jit.script
def _linear_weight_norm_sample_non_sequential_ghostnorm_T(input, grad_output, grad_norm2):
    """The Goodfellow trick, i.e., Frobenius norm equal to product of 2-norms."""
    grad_norm2 = grad_norm2 + (input.norm(2, dim=1) * grad_output.norm(2, dim=1))**2
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    grad_output = torch.einsum('B,B...->B...',clip_factor,grad_output)
    return torch.einsum('B...p,B...d->dp',grad_output,input), clip_factor

@torch.jit.script
def _linear_weight_norm_sample_psg(input, grad_output, grad_norm2):
    grad_weight=torch.einsum('B...d,B...p->Bpd', input, grad_output)
    grad_norm2 = grad_norm2 + torch.sum(grad_weight**2, dim=(1, 2))
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    return torch.einsum('B,B...->...',clip_factor,grad_weight), clip_factor

@torch.jit.script
def _linear_weight_norm_sample_psg_T(input, grad_output, grad_norm2):
    grad_weight=torch.einsum('B...d,B...p->Bdp', input, grad_output)
    grad_norm2 = grad_norm2 + torch.sum(grad_weight**2, dim=(1, 2))
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    return torch.einsum('B,B...->...',clip_factor,grad_weight), clip_factor


def sum_over_all_but_batch_and_last_n(tensor: torch.Tensor, n_dims: int) -> torch.Tensor:
    if tensor.dim() == n_dims + 1:
        return tensor
    else:
        dims = list(range(1, tensor.dim() - n_dims))
        return tensor.sum(dim=dims)

@torch.jit.script
def _layer_norm_grad_sample_same_dim(input, grad_norm2, grad_weight):
    grad_norm2 = grad_norm2 + grad_weight.flatten(start_dim=1).norm(2, dim=1)**2
    clip_factor = 1/(torch.sqrt(grad_norm2.to(input))+1e-4)
    return torch.einsum('B,B...->...',clip_factor,grad_weight), clip_factor

class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        return F.linear(input,weight,bias)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        
        #-- inherit, initialize
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum('B...p,pd->B...d',grad_output,weight)
            
        if hasattr(weight,'ds_numel'):
            weight.numels=weight.ds_numel
        else:
            weight.numels=weight.numel()

        #-- ghost norm for weight, per-sample grad instantiation for bias
        if bias is not None and ctx.needs_input_grad[2]:
            if bias.per_sample_clip==0:
                grad_bias = torch.einsum('B...p->p',grad_output) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                #grad_bias.nan_to_num_()
            else:
                grad_bias = torch.einsum('B...p->Bp',grad_output)
                grad_norm2=grad_bias.norm(2, dim=1)**2
        else:
            grad_norm2=torch.zeros(grad_output.shape[0],device=grad_output.device)

        if ctx.needs_input_grad[1]:
            if weight.per_sample_clip==0:
                grad_weight=torch.einsum('B...d,B...p->pd', input, grad_output) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight.nan_to_num_()
            else:

                T = math.prod(list(grad_output.shape[1:-1]))
                if 2*(T**2) > weight.numels:
                    grad_weight, clip_factor=_linear_weight_norm_sample_psg(input,grad_output,grad_norm2)
                elif input.dim()==3:
                    grad_weight, clip_factor=_linear_weight_norm_sample_sequential_ghostnorm(input,grad_output,grad_norm2)
                else:
                    grad_weight, clip_factor=_linear_weight_norm_sample_non_sequential_ghostnorm(input,grad_output,grad_norm2)
                grad_weight = grad_weight / math.sqrt(weight.n_layers) 
                grad_weight = grad_weight + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = torch.einsum('B,B...->...',clip_factor,grad_bias) / math.sqrt(bias.n_layers) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
        return grad_input, grad_weight, grad_bias

class Conv2dFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, kernel_size=None):
        ctx.save_for_backward(input,weight, bias)
        #https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/2
        ctx.stride=stride 
        ctx.padding=padding 
        ctx.dilation=dilation 
        ctx.groups=groups
        ctx.kernel_size=kernel_size

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        
        #-- inherit, initialize
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding=ctx.padding 
        dilation=ctx.dilation
        groups=ctx.groups
        kernel_size=ctx.kernel_size

        grad_input = grad_weight = grad_bias = None        

        # nn.grad is specifically for convolution layers!
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)

        #-- ghost norm for weight, per-sample grad instantiation for bias
        grad_output_flattened = grad_output.flatten(2).transpose(-1,-2)        # F^{-1}(dL/ds); shape B*p*T->B*T*p
        
        if bias is not None and ctx.needs_input_grad[2]:
            if bias.per_sample_clip==0:
                grad_bias = grad_output_flattened.sum(dim=(0,1))+ torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
            else:
                grad_bias = grad_output_flattened.sum(dim=1)
                grad_norm2=grad_bias.norm(2, dim=1)**2
        else:
            grad_norm2=torch.zeros(grad_output_flattened.shape[0],device=grad_output.device)

        T = grad_output_flattened.shape[1]

        if ctx.needs_input_grad[1]:
            input_unfolded = F.unfold(input, kernel_size=kernel_size,
                                    dilation=dilation, padding=padding,
                                    stride=stride).transpose(-1,-2) # U(a); shape B*D*T->B*T*D
            if weight.per_sample_clip==0:
                grad_weight=torch.einsum('B...d,B...p->pd', input_unfolded, grad_output_flattened)
                grad_weight = grad_weight.view(-1, *weight.shape)[0] + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
            else:
                if 2*(T**2) > weight.numels:
                    grad_weight, clip_factor=_linear_weight_norm_sample_psg(input_unfolded,grad_output_flattened,grad_norm2)
                else:
                    grad_weight, clip_factor=_linear_weight_norm_sample_sequential_ghostnorm(input_unfolded,grad_output_flattened,grad_norm2)
                grad_weight = grad_weight.view(-1, *weight.shape)[0] / math.sqrt(weight.n_layers) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups) 
                #grad_weight.nan_to_num_()
    
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = torch.einsum('B,B...->...',clip_factor,grad_bias) / math.sqrt(bias.n_layers)+ torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                    #grad_bias.nan_to_num_()
                    #print('rank: ', torch.distributed.get_rank(),torch.randn(1))
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

class GroupNormFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, num_groups, weight=None, bias=None, eps=1e-05):
        ctx.save_for_backward(input, weight, bias)
        #We have several things to save to accelerate computation

        #https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/2
        ctx.num_groups=num_groups 
        ctx.eps=eps 
        return F.group_norm(input, num_groups, weight, bias, eps)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        
        #-- inherit, initialize
        input, weight, bias = ctx.saved_tensors
        num_groups= ctx.num_groups
        eps=ctx.eps
        input_size = input.size()
        input_hat = F.group_norm(input, num_groups, eps=eps).view(input_size[0], num_groups, -1)

        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            var = input.view(input_size[0], num_groups, -1).var(-1, keepdim = True, unbiased = False)
            dL_dx_hat = (torch.einsum('ji...,i->ji...',grad_output,weight)).view(input_size[0], num_groups, -1)
            grad_input = ((dL_dx_hat -dL_dx_hat.mean(-1, keepdim = True)) -  input_hat*((input_hat*dL_dx_hat).mean(-1, keepdim = True)))/(var+eps).sqrt()
            grad_input = grad_input.view(input_size)
            del var
            del dL_dx_hat


        #-- per-sample grad instantiation for weight & bias
        if bias is not None and ctx.needs_input_grad[3]:
            if bias.per_sample_clip==0:
                grad_bias = torch.einsum('bi...->i',grad_output) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                #grad_bias.nan_to_num_()
            else:
                grad_bias = torch.einsum('bi...->bi',grad_output)
                grad_norm2 = grad_bias.flatten(start_dim=1).norm(2, dim=1)**2
        else:
            grad_norm2=torch.zeros(grad_output.shape[0],device=grad_output.device)

        if ctx.needs_input_grad[2]:
            if weight.per_sample_clip==0:
                grad_weight = torch.einsum('bi...->i',F.group_norm(input, num_groups, eps=eps) * grad_output) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight.nan_to_num_()
            else:                
                grad_weight = torch.einsum('bi...->bi',input_hat.view(input_size) * grad_output)
                grad_norm2 += grad_weight.norm(2, dim=1)**2
                #del input_hat
                #del input_size
        
                #-- clip factor is used to reweight per-sample grad
                clip_factor = 1/(torch.sqrt(grad_norm2)+1e-4)
                grad_output = torch.einsum('B,B...->B...',clip_factor,grad_output)
                
                #-- generate grad as standard (but now grad_output is reweighted)
                grad_weight = torch.einsum('bi...->i',F.group_norm(input, num_groups, eps=eps) * grad_output) / math.sqrt(weight.n_layers) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight.nan_to_num_()
                if bias is not None and ctx.needs_input_grad[3]:
                    grad_bias = torch.einsum('B,B...->...',clip_factor,grad_bias) / math.sqrt(bias.n_layers) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                    #grad_bias.nan_to_num_()

        return grad_input, None, grad_weight, grad_bias, None

class EmbeddingFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
        ctx.save_for_backward(input,weight)
        #https://discuss.pytorch.org/t/how-to-save-a-list-of-integers-for-backward-when-using-cpp-custom-layer/25483/2
        ctx.padding_idx=padding_idx 
        ctx.max_norm=max_norm 
        ctx.norm_type=norm_type 
        ctx.scale_grad_by_freq=scale_grad_by_freq 
        ctx.sparse=sparse 

        return F.embedding(input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        
        #-- inherit, initialize
        input, weight = ctx.saved_tensors
        padding_idx= ctx.padding_idx


        grad_weight = None

        if ctx.needs_input_grad[1]:
            if weight.per_sample_clip==0:
                grad_output=grad_output.view(-1, weight.shape[1])
                input=input.view(-1)
                grad_weight = torch.zeros_like(weight).to(grad_output)
                grad_weight= grad_weight.index_add(0, input, grad_output) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight.nan_to_num_()
            else:        
                #-- ghost norm for weight, no bias
                #--- compute gradient norm
                not_AAt: torch.Tensor = ~input[:, :, None].eq(input[:, None, :])
                # Clear the contribution to the norm of the gradient for the padding token.
                #   In vanilla backpropagation, this particular embedding doesn't contribute to the gradient anyway.
                #   For more see 1.10.0 doc: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
                #       'the embedding vector at padding_idx is not updated during training, i.e. it remains as a fixed “pad”.'
                if padding_idx is not None:
                    # The right way to think about the next line of code is that A_i[t, padding_idx] = 0 for all t in [T].
                    #   So the entry gets cleared whenever one of A, A^t takes the padding idx.
                    not_AAt.bitwise_or_((input[:, :, None] == padding_idx) | (input[:, None, :] == padding_idx))
                grad_norm = torch.sqrt((torch.bmm(grad_output, grad_output.transpose(-1, -2)).masked_fill(not_AAt, 0)).sum(dim=(1, 2)))
        
                #-- clip factor is used to reweight per-sample grad
                clip_factor = 1/(grad_norm+1e-4)
                grad_output = torch.einsum('B,B...->B...',clip_factor,grad_output)
                
                #-- generate grad as standard (but now grad_output is reweighted)
                ### Option 1: look up
                grad_output=grad_output.view(-1, weight.shape[1])
                input=input.view(-1)
                grad_weight = torch.zeros_like(weight).to(grad_output)
                grad_weight= grad_weight.index_add(0, input, grad_output) / math.sqrt(weight.n_layers) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                '''
                ### Option 2: matrix multiplication
                input = F.one_hot(input, num_classes=weight.shape[0]).to(grad_output)  # (batch_size, seq_len, vocab_dim,)
                grad_weight = torch.einsum('...p,...d->dp',grad_output,input)+ torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                '''
    
                # `torch.nn.Embedding` layers don't accumulate gradient on the padding_idx position.
                if padding_idx is not None:
                    # `grad_weight` has size (batch_size, num_vocab, embedding_dim).
                    grad_weight[:, padding_idx, :] = 0.
                #grad_weight.nan_to_num_()

        return None, grad_weight, None, None, None, None, None
    

 
class LayerNormFunction(Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, normalized_shape, weight, bias=None, eps = 1e-5):
        
        ctx.normalized_shape=normalized_shape 
        ctx.eps=eps
        
        # https://github.com/NVIDIA/apex/blob/136a13cb67b853181a4bf552688787450f4e8fbe/apex/normalization/fused_layer_norm.py#L122-L142

        #fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        #output, mean, variance = fused_layer_norm_cuda.forward_affine(
        #    input, normalized_shape, weight, bias, eps)
        
        ctx.save_for_backward(input, weight, bias)
        
        return F.layer_norm(input,normalized_shape,weight, bias, eps)

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        normalized_shape= ctx.normalized_shape
        eps=ctx.eps

        grad_input = grad_weight = grad_bias = None
        
        
        #-- inherit, initialize
        input, weight, bias = ctx.saved_tensors
        
        if ctx.needs_input_grad[0]:
            # xi_hat=(input - mean)/variance+eps
            to_normalize_dim=tuple(range(-len(normalized_shape),0))
            mean=torch.mean(input,dim=to_normalize_dim,keepdim=True)
            variance = torch.var(input, dim=to_normalize_dim, unbiased=False, keepdim=True)+eps

            B = math.prod(normalized_shape)
            dL_dxi_hat = grad_output * weight # https://pytorch.org/docs/stable/notes/broadcasting.html
            dL_dvar = (dL_dxi_hat * (-0.5) *  (input - mean)  * (variance ** -1.5)).sum(dim=to_normalize_dim, keepdim=True)
            dL_dmean = (dL_dxi_hat * (-1.0 / torch.sqrt(variance)) + dL_dvar * (-2.0 * (input - mean)) / B).sum(dim=to_normalize_dim, keepdim=True)
            # notice that var is defined via mean
            
            grad_input = (dL_dxi_hat / torch.sqrt(variance)) + (dL_dvar * 2.0 *  (input - mean) / B) + (dL_dmean / B)
        
        #-- per-sample grad instantiation for weight & bias
        if bias is not None and ctx.needs_input_grad[3]:
            if bias.per_sample_clip==0:
                grad_bias = sum_over_all_but_batch_and_last_n(grad_output, bias.dim()).sum(dim=0) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                #grad_bias.nan_to_num_()
            else:
                grad_bias = sum_over_all_but_batch_and_last_n(grad_output, bias.dim())
                grad_norm2 = grad_bias.flatten(start_dim=1).norm(2, dim=1)**2
        else:
            grad_norm2=torch.zeros(grad_output.shape[0],device=grad_output.device)

        if ctx.needs_input_grad[2]:
            grad_weight = F.layer_norm(input, normalized_shape, eps=eps) * grad_output
            if grad_output.dim() != weight.dim() + 1:
                dims = list(range(1, grad_output.dim() - weight.dim()))
                grad_weight = grad_weight.sum(dim=dims)
                
            if weight.per_sample_clip==0:
                grad_weight = grad_weight.sum(dim=0)+ torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight.nan_to_num_()
            else:    
                grad_weight, clip_factor = _layer_norm_grad_sample_same_dim(input, grad_norm2, grad_weight)
                grad_weight = grad_weight / math.sqrt(weight.n_layers) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                #grad_weight.nan_to_num_()

                if bias is not None and ctx.needs_input_grad[3]:
                    grad_bias = torch.einsum('B,B...->...',clip_factor,grad_bias) / math.sqrt(bias.n_layers) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                    #grad_bias.nan_to_num_()

        return grad_input, None, grad_weight, grad_bias, None

  
class transformerConv1DFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        #https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L82
        size_out = input.size()[:-1] + (weight.shape[1],)
        output = torch.addmm(bias, input.view(-1, input.size(-1)), weight)
        return output.view(size_out)
        ## weight transpose is extremely expensive!!!
        # return F.linear(input,weight.t(),bias) #

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):        
        #-- inherit, initialize
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum('B...p,dp->B...d',grad_output,weight)
            
        #-- ghost norm for weight, per-sample grad instantiation for bias
        if bias is not None and ctx.needs_input_grad[2]:
            if bias.per_sample_clip==0:
                grad_bias = torch.einsum('B...p->p',grad_output) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
                #grad_bias.nan_to_num_()
            else:
                grad_bias = torch.einsum('B...p->Bp',grad_output)
                grad_norm2=grad_bias.norm(2, dim=1)**2
        else:
            grad_norm2=torch.zeros(grad_output.shape[0],device=grad_output.device)

        if ctx.needs_input_grad[1]:
            if weight.per_sample_clip==0:
                grad_weight=torch.einsum('B...d,B...p->dp', input, grad_output) + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
            else:
                T = math.prod(list(grad_output.shape[1:-1]))
                if 2*(T**2) > weight.numels:
                    grad_weight, clip_factor=_linear_weight_norm_sample_psg_T(input,grad_output,grad_norm2)
                elif input.dim()==3:
                    grad_weight, clip_factor=_linear_weight_norm_sample_sequential_ghostnorm_T(input,grad_output,grad_norm2)
                else:
                    grad_weight, clip_factor=_linear_weight_norm_sample_non_sequential_ghostnorm_T(input,grad_output,grad_norm2)
                grad_weight = grad_weight / math.sqrt(weight.n_layers) 
                grad_weight = grad_weight + torch.randn_like(weight, device= weight.device, dtype= weight.dtype)* weight.noise
                if bias is not None and ctx.needs_input_grad[2]:
                    grad_bias = torch.einsum('B,B...->...',clip_factor,grad_bias) / math.sqrt(bias.n_layers) + torch.randn_like(bias, device= bias.device, dtype= bias.dtype)* bias.noise
        return grad_input, grad_weight, grad_bias



#%% extending nn.Module with per-sample gradient clipping, through https://pytorch.org/docs/stable/notes/extending.html

# STEPS: copy nn.XX (e.g. Linear) module, re-write forward with our customized functional


#https://pytorch.org/docs/stable/notes/extending.html#adding-a-module
class DPLinear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        #print(LinearFunction.apply(input, self.weight, self.bias)[0][0])
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class DPtransformersConv1D(Module):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py#L82

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DPtransformersConv1D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs))
        self.bias = Parameter(torch.zeros(out_features))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, input: Tensor) -> Tensor:
        return transformerConv1DFunction.apply(input, self.weight, self.bias)

import numbers
_shape_t = Union[int, List[int], Size]

class DPLayerNorm(Module):
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DPLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return LayerNormFunction.apply(input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
   

class DPGroupNorm(Module):
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return GroupNormFunction.apply(input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
 
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd

class DPConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=None,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(DPConv2d, self).__init__(in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

        self.kernel_size=kernel_size #!!!

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return Conv2dFunction.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups, self.kernel_size)
        
        #print(Conv2dFunction.apply(input, weight, bias, self.stride,
        #                self.padding, self.dilation, self.groups, self.kernel_size)[0][0][0])
        return Conv2dFunction.apply(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups, self.kernel_size)
        
    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


class DPEmbedding(Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(DPEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if _weight is None:
            self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = Parameter(_weight)

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return EmbeddingFunction.apply(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
#%% iteatively replacing nn.Module by our customized modules, must inherit the parameters!!
from .autograd_grad_sample_dist import requires_grad
#https://discuss.pytorch.org/t/replacing-convs-modules-with-custom-convs-then-notimplementederror/17736
def replace_Linear(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) == torch.nn.Linear and requires_grad(layer):
            new_layer = DPLinear(in_features=layer.in_features, out_features=layer.out_features, 
                               bias=(layer.bias!=None), device=layer.weight.device, dtype=layer.weight.dtype)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            del layer

            setattr(module, layer_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()

    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_Linear(immediate_child_module)

def replace_transformersConv1D(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) == transformers.pytorch_utils.Conv1D and requires_grad(layer):
            new_layer = DPtransformersConv1D(in_features=layer.weight.shape[0], out_features=layer.weight.shape[1], 
                               device=layer.weight.device, dtype=layer.weight.dtype)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            del layer
            setattr(module, layer_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()

    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_transformersConv1D(immediate_child_module)

def replace_LayerNorm(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) == torch.nn.LayerNorm and requires_grad(layer):
            new_layer = DPLayerNorm(normalized_shape=layer.normalized_shape, eps=layer.eps, 
                                  elementwise_affine=layer.elementwise_affine,
                                  device=layer.weight.device, dtype=layer.weight.dtype)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            del layer
            setattr(module, layer_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_LayerNorm(immediate_child_module)


def replace_GroupNorm(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) == torch.nn.GroupNorm and requires_grad(layer):
            new_layer = DPGroupNorm(num_groups = layer.num_groups, num_channels = layer.num_channels, eps = layer.eps, affine = layer.affine, device = layer.weight.device, dtype = layer.weight.dtype)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            del layer
            setattr(module, layer_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_GroupNorm(immediate_child_module)

def replace_Conv2d(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) == torch.nn.Conv2d and requires_grad(layer):
            new_layer = DPConv2d(in_channels=layer.in_channels, out_channels=layer.out_channels, 
                               kernel_size=layer.kernel_size, stride=layer.stride, 
                               padding=layer.padding, dilation=layer.dilation, 
                               groups=layer.groups, bias=(layer.bias!=None), 
                               padding_mode=layer.padding_mode,
                               device=layer.weight.device, dtype=layer.weight.dtype)
            new_layer.weight = layer.weight
            new_layer.bias = layer.bias
            del layer
            setattr(module, layer_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_Conv2d(immediate_child_module)
            

def replace_Embedding(module):
    for layer_str in dir(module):
        layer = getattr(module, layer_str)
        if type(layer) == torch.nn.Embedding and requires_grad(layer):
            new_layer = DPEmbedding(num_embeddings=layer.num_embeddings, 
                                    embedding_dim=layer.embedding_dim, 
                                    padding_idx=layer.padding_idx, 
                                    max_norm=layer.max_norm, norm_type=layer.norm_type, 
                                    scale_grad_by_freq=layer.scale_grad_by_freq, sparse=layer.sparse, 
                                    device=layer.weight.device, dtype=layer.weight.dtype)
                                    
            new_layer.weight = layer.weight
            del layer
            setattr(module, layer_str, new_layer)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    if hasattr(module,'children'):
        for immediate_child_module in module.children():
            replace_Embedding(immediate_child_module)        
