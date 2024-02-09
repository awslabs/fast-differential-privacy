'''Train CV with PyTorch.'''
def main(args):

    device= torch.device("cuda:0")

    # Data
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224,224)),#https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/10
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])

    if args.dataset_name in ['SVHN','CIFAR10']:
        num_classes=10
    elif args.dataset_name in ['CIFAR100','FGVCAircraft']:
        num_classes=100
    elif args.dataset_name in ['Food101']:
        num_classes=101
    elif args.dataset_name in ['GTSRB']:
        num_classes=43
    elif args.dataset_name in ['CelebA']:
        num_classes=40
    elif args.dataset_name in ['Places365']:
        num_classes=365
    elif args.dataset_name in ['ImageNet']:
        num_classes=1000
    elif args.dataset_name in ['INaturalist']:
        num_classes=10000
        

    if args.dataset_name in ['SVHN','Food101','GTSRB','FGVCAircraft']:
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='train', download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='test', download=True, transform=transformation)
    elif args.dataset_name in ['CIFAR10','CIFAR100']:
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', train=True, download=True, transform=transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', train=False, download=True, transform=transformation)
    elif args.dataset_name=='CelebA':
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='train', download=False, target_type='attr', transform=transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='test', download=False, target_type='attr',transform=transformation)
    elif args.dataset_name=='Places365':
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='train-standard', small=True, download=False, transform=transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='val', small=True, download=False, transform=transformation)
    elif args.dataset_name=='INaturalist':
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', version='2021_train_mini', download=False, transform=transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', version='2021_valid', download=False, transform=transformation)
    elif args.dataset_name=='ImageNet':
        trainset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='train', transform=transformation)
        testset = getattr(torchvision.datasets,args.dataset_name)(root='data/', split='val', transform=transformation)
        
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    n_acc_steps = args.bs // args.mini_bs # gradient accumulation steps


    # Model
    net = timm.create_model(args.model, pretrained=True, num_classes = num_classes)
    net = ModuleValidator.fix(net).to(device)

    if args.dataset_name=='CelebA':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()
    

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if 'BiTFiT' in args.clipping_mode:
        for name,layer in net.named_modules():
            if hasattr(layer,'weight'):
                temp_layer=layer
        for name,param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)
        for param in temp_layer.parameters():
            param.requires_grad_(True)

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        sigma=get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 1/len(trainset),
                sample_rate = args.bs/len(trainset),
                epochs = args.epochs,
            )
        print(f'adding noise level {sigma}')
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(trainset),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode='MixOpt',
            clipping_style='all-layer',
        )
        privacy_engine.attach(optimizer)        

        
    tr_loss=[]
    te_loss=[]
    tr_acc=[]
    te_acc=[]
    
    def train(epoch):

        net.train()
        train_loss = 0
        correct = 0
        total = 0

   
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if args.dataset_name=='CelebA':
                loss = criterion(outputs, targets.float()).sum(dim=1).mean()
            else:
                loss = criterion(outputs, targets);#print(loss.item())

            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()     

            train_loss += loss.item()
            total += targets.size(0)
            if args.dataset_name=='CelebA':
                correct += ((outputs > 0) == targets).sum(dim=0).float().mean()
            else:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            if args.dataset_name in ['Places365','INaturalist','ImageNet'] and (batch_idx + 1) % 100 == 0:
                print(loss.item(),100.*correct/total)

                
        tr_loss.append(train_loss/(batch_idx+1))
        tr_acc.append(100.*correct/total)
        print('Epoch: ', epoch, 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                if args.dataset_name=='CelebA':
                    loss = criterion(outputs, targets.float()).sum(dim=1).mean()
                else:
                    loss = criterion(outputs, targets);#print(loss.item())

                test_loss += loss.item()
                total += targets.size(0)
                if args.dataset_name=='CelebA':
                    correct += ((outputs > 0) == targets).sum(dim=0).float().mean()
                else:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()

            te_loss.append(test_loss/(batch_idx+1))
            te_acc.append(100.*correct/total)
            print('Epoch: ', epoch, 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
    print(tr_loss,tr_acc,te_loss,te_acc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--epochs', default=5, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--mini_bs', type=int, default=100)
    parser.add_argument('--epsilon', default=8, type=float, help='target epsilon')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10',help='https://pytorch.org/vision/stable/datasets.html')
    parser.add_argument('--clipping_mode', type=str, default='MixOpt',choices=['BiTFiT','MixOpt', 'nonDP','nonDP-BiTFiT'])
    parser.add_argument('--model', default='vit_base_patch16_224', type=str, help='model name')

    args = parser.parse_args()
    
    from fastDP import PrivacyEngine

    import torch
    import torchvision
    torch.manual_seed(2)
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from opacus.validators import ModuleValidator
    from opacus.accountants.utils import get_noise_multiplier
    from tqdm import tqdm
    import numpy as np
    import warnings; warnings.filterwarnings("ignore")
    main(args)
