import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from fastDP import PrivacyEngine_Distributed_extending

import timm
#from opacus.validators import ModuleValidator
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")


import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from fairscale.nn import FullyShardedDataParallel as FSDP

#--- if import from torch <= 1.11
#from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload,BackwardPrefetch
#from torch.distributed.fsdp.wrap import default_auto_wrap_policy,enable_wrap,wrap
from fairscale.nn import default_auto_wrap_policy
from fairscale.internal.parallel import ProcessGroupName


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    

def train(epoch,net,rank,trainloader,criterion,optimizer,grad_acc_steps):
    net.train()
    ddp_loss = torch.zeros(3).to(rank)
   
    for batch_idx, data in enumerate(tqdm(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data[0].to(rank), data[1].to(rank)
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        loss.backward()
        if ((batch_idx + 1) % grad_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()
            
        _, predicted = outputs.max(1)

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data[0])
        ddp_loss[2] += predicted.eq(targets.view_as(predicted)).sum().item()

    if rank == 0:
        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%%'
              % (ddp_loss[0]/(batch_idx+1), 100.*ddp_loss[2]/ddp_loss[1]))
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

def test(epoch,net,rank,testloader,criterion):
    net.eval()
    ddp_loss = torch.zeros(3).to(rank)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(testloader)):
            inputs, targets = data[0].to(rank), data[1].to(rank)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            _, predicted = outputs.max(1)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(data[0])
            ddp_loss[2] += predicted.eq(targets.view_as(predicted)).sum().item()
    if rank == 0:
        print('Epoch: ', epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%%'
              % (ddp_loss[0]/ddp_loss[1]*len(inputs), 100.*ddp_loss[2]/ddp_loss[1]))

'''Train CIFAR10/CIFAR100 with PyTorch.'''
def main(rank, world_size, args):

    grad_acc_steps = args.batch_size//args.mini_batch_size//world_size

    if args.clipping_mode not in ['nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT']:
        print("Mode must be one of 'nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT'")
        return None

    setup(rank, world_size)


    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.dimension),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    # Data
    print('==> Preparing data..')

    if args.cifar_data=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=False, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=False, transform=transformation)
    elif args.cifar_data=='CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=False, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=False, transform=transformation)
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100"
         
    sampler_train = DistributedSampler(trainset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler_test = DistributedSampler(testset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.mini_batch_size, 'sampler': sampler_train}
    test_kwargs = {'batch_size': 10, 'sampler': sampler_test}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': False,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    trainloader = torch.utils.data.DataLoader(trainset,**train_kwargs)
    testloader = torch.utils.data.DataLoader(testset, **test_kwargs)
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Model
    print('==> Building and fixing model..', args.model,'. Mode: ', args.clipping_mode,grad_acc_steps)
    net = timm.create_model(args.model, pretrained=True, num_classes=int(args.cifar_data[5:]))    
  
    if 'BiTFiT' in args.clipping_mode:
      for name,param in net.named_parameters():
          if '.bias' not in name:
              param.requires_grad_(False)
      
    net = net.to(rank)

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        PrivacyEngine_Distributed_extending(
            net,
            batch_size=args.batch_size,
            sample_size=len(trainset),
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            num_GPUs=world_size,
            torch_seed_is_fixed=True, #FSDP always gives different seeds to devices if use FSDP() to wrap
            grad_accum_steps=grad_acc_steps,
        )


    #net = FSDP(net,flatten_parameters=False, mixed_precision=args.fp16)# must use flatten_parameters=False https://github.com/facebookresearch/fairscale/issues/1047
    
    from fairscale.nn.wrap import wrap, enable_wrap, auto_wrap
    fsdp_params = dict(wrapper_cls=FSDP, mixed_precision=args.fp16, flatten_parameters=False)#,disable_reshard_on_root=False,reshard_after_forward=False,clear_autocast_cache=True) # True or False
    with enable_wrap(**fsdp_params):
        # cannot wrap the network as a whole, will lose weight.noise
        for pp in net.modules(): # must wrap module/layer not parameter
            if hasattr(pp,'weight'): # AssertionError assert not isinstance(child, cast(type, ConfigAutoWrap.wrapper_cls))
                pp=auto_wrap(pp)
    
    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    

    criterion = nn.CrossEntropyLoss(reduction='sum')
  
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    #https://pytorch.org/docs/stable/fsdp.html
    #The optimizer must be initialized after the module has been wrapped, since FSDP will shard parameters in-place and this will break any previously initialized optimizers.

    init_start_event.record()
    
    for epoch in range(args.epochs):
        train(epoch,net,rank,trainloader,criterion,optimizer,grad_acc_steps)
        test(epoch,net,rank,testloader,criterion)
    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        
    cleanup()
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='numter of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='logical batch size')
    parser.add_argument('--mini_batch_size', default=16, type=int, help='physical batch size')
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--model', default='vit_gigantic_patch14_224', type=str)
    parser.add_argument('--cifar_data', type=str, default='CIFAR100')
    parser.add_argument('--dimension', type=int,default=224)
    parser.add_argument('--fp16', type=bool, default=False)

    args = parser.parse_args()
    
    torch.manual_seed(2) # useful for reproduction

    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(main,args=(WORLD_SIZE, args),
            nprocs=WORLD_SIZE,join=True)
    #https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn
