'''Two-phase Training CIFAR10/CIFAR100'''
def main(args):

    device= torch.device("cuda:0")

    # Data
    print('==> Preparing data..')

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5)),
    ])


    if args.cifar_data=='CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR10(root='data/', train=False, download=True, transform=transformation)
    elif args.cifar_data=='CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='data/', train=True, download=True, transform=transformation)
        testset = torchvision.datasets.CIFAR100(root='data/', train=False, download=True, transform=transformation)
    else:
        return "Must specify datasets as CIFAR10 or CIFAR100"
         
 
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.mini_bs, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)

    n_acc_steps = args.bs // args.mini_bs # gradient accumulation steps

    # Model
    print('==> Building model..', args.model,'; BatchNorm is replaced by GroupNorm.')
    net = timm.create_model(args.model,pretrained=True,num_classes=int(args.cifar_data[5:]))    
    net = ModuleValidator.fix(net).to(device)

    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr_full)


    # Privacy engine
    sigma=get_noise_multiplier(
            target_epsilon = args.epsilon,
            target_delta = 1e-5,
            sample_rate = args.bs/len(trainset),
            epochs = args.epochs,
            accountant = "rdp",
        )
    privacy_engine = PrivacyEngine(
        net,
        batch_size=args.bs,
        sample_size=len(trainset),
        noise_multiplier=sigma,
        epochs=args.epochs,
        clipping_mode='MixOpt',
    )
    privacy_engine.attach(optimizer)        

        
    def train(epoch):

        net.train()
        train_loss = 0
        correct = 0
        total = 0

   
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()                    
            else:
                # accumulate per-example gradients but don't take a step yet
                optimizer.virtual_step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Epoch: ', epoch, len(trainloader), 'Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Epoch: ', epoch, len(testloader), 'Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    for epoch in range(args.epochs):
        #----------------- two-phase training
        if epoch==args.mix_epoch:
            privacy_engine.detach()  

            for name,param in net.named_parameters():
                if '.bias' not in name:
                    param.requires_grad=False

            optimizer = optim.Adam(net.parameters(), lr=args.lr_BiTFiT)

            privacy_engine = PrivacyEngine(
                net,
                batch_size=args.bs,
                sample_size=len(trainset),
                noise_multiplier=sigma,
                epochs=args.epochs,
                clipping_mode='ghost',
            )
            privacy_engine.attach(optimizer)
        #----------------------------
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr_full', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--lr_BiTFiT', default=0.005, type=float, help='learning rate')
    parser.add_argument('--epochs', default=3, type=int,
                        help='numter of epochs')
    parser.add_argument('--bs', default=1000, type=int, help='batch size')
    parser.add_argument('--epsilon', default=2, type=float, help='target epsilon')
    parser.add_argument('--model', default='vit_small_patch16_224', type=str)
    parser.add_argument('--mini_bs', type=int, default=50)
    parser.add_argument('--cifar_data', type=str, default='CIFAR10')
    parser.add_argument('--mix_epoch', type=int, default=1)

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
    import warnings; warnings.filterwarnings("ignore")

    main(args)
