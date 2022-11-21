#This runs multi-label classification
def main(args):
    if args.clipping_mode not in ['nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT']:
        print("Mode must be one of 'nonDP','BK-ghost', 'BK-MixGhostClip', 'BK-MixOpt','nonDP-BiTFiT','BiTFiT'")
        return None

    device = torch.device('cuda')

    # Data
    print('==> Preparing data..')
    
    train_set = datasets.CelebA(root='.', split='train',target_type='attr',
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
                                    ]))
    test_set = datasets.CelebA(root=".", split='test', target_type='attr',
        transform=transforms.Compose([
           transforms.ToTensor()]))

    if args.labels==None:
        args.labels=list(range(40))
        print('Training on all 40 labels.')
    else:
        print('Training on ', [attr_names[ind] for ind in args.labels])
        
        
    train_set.attr = train_set.attr[:, args.labels].type(torch.float32)
    test_set.attr = test_set.attr[:, args.labels].type(torch.float32)

    print('Training/Testing set size: ', len(train_set),len(test_set),' ; Image dimension: ',train_set[0][0].shape)

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.mini_bs, pin_memory=True,num_workers=4,shuffle=True)
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=500, pin_memory=True,num_workers=4, shuffle=False)

    n_acc_steps=args.bs//args.mini_bs

    # Model
    print('==> Building model..', args.model,'; BatchNorm is replaced by GroupNorm.')
    net = timm.create_model(args.model, pretrained=True, num_classes=len(args.labels))
    net = ModuleValidator.fix(net)
    net=net.to(device)
    
    for name,param in net.named_parameters():
        print("First trainable parameter is: ",name);break
    
    print('Number of total parameters: ', sum([p.numel() for p in net.parameters()]))
    print('Number of trainable parameters: ', sum([p.numel() for p in net.parameters() if p.requires_grad]))
    
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    if 'BiTFiT' in args.clipping_mode:
        for name,param in net.named_parameters():
            if '.bias' not in name:
                param.requires_grad_(False)

    # Privacy engine
    if 'nonDP' not in args.clipping_mode:
        sigma=get_noise_multiplier(
                target_epsilon = args.epsilon,
                target_delta = 5e-6,
                sample_rate = args.bs/len(train_set),
                epochs = args.epochs,
            )

        if 'BK' in args.clipping_mode:
            clipping_mode=args.clipping_mode[3:]
        else:
            clipping_mode='ghost'
        privacy_engine = PrivacyEngine(
            net,
            batch_size=args.bs,
            sample_size=len(train_set),
            noise_multiplier=sigma,
            epochs=args.epochs,
            clipping_mode=clipping_mode,
            origin_params=args.origin_params,
            clipping_style=args.clipping_style,
        )
        privacy_engine.attach(optimizer)   
        
        
    def train(epoch):

        net.train()
        train_loss = 0
        correct = np.zeros_like([0]*len(args.labels))
        total = 0
        for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets.float()).sum(dim=1).mean()


            loss.backward()
            if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(trainloader)):
                optimizer.step()
                optimizer.zero_grad()                

            train_loss += loss.item()
            total += targets.size(0)
            correct += ((outputs > 0) == targets).sum(dim=0).cpu().detach().numpy()

        print('Epoch: ', epoch, 'Train Loss: ', (train_loss/(batch_idx+1), 
                ' | Acc: ', 100.*correct/total, np.mean(100.0 * correct / total)))

    def test(epoch):
        net.eval()
        test_loss = 0
        correct = np.zeros_like([0]*len(args.labels))
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(testloader)):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets.float()).sum(dim=1)
                loss = loss.mean()

                test_loss += loss.item()
                total += targets.size(0)
                correct += ((outputs > 0) == targets).sum(dim=0).cpu().detach().numpy()

            print('Epoch: ', epoch, 'Test Loss: ', (test_loss/(batch_idx+1), 
                    ' | Acc: ', 100.*correct/total, np.mean(100.0 * correct / total)))


    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs', type=int, default=500)
    parser.add_argument('--mini_bs', type=int, default=100)
    parser.add_argument('--epsilon', default=3, type=float)
    parser.add_argument('--clipping_mode', default='BK-MixOpt', type=str)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--labels', nargs="*", type=int, default=None,help='List of label indices, 0-39 for CelebA')
    parser.add_argument('--origin_params', nargs='+', default=None)
    parser.add_argument('--clipping_style', type=str, default='all-layer')

    
    args = parser.parse_args()

    attr_names=['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes',
            'Bald','Bangs','Big_Lips','Big_Nose',
            'Black_Hair','Blond_Hair','Blurry','Brown_Hair',
            'Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses',
            'Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones',
            'Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes',
            'No_Beard','Oval_Face','Pale_Skin','Pointy_Nose',
            'Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling',
            'Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat',
            'Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']
    
    import numpy as np
    from fastDP import PrivacyEngine

    import torch
    from torchvision import datasets, transforms
    torch.manual_seed(0)
    import torch.nn as nn
    import torch.optim as optim
    import timm
    from opacus.validators import ModuleValidator
    from opacus.accountants.utils import get_noise_multiplier
    from tqdm import tqdm
    import warnings; warnings.filterwarnings("ignore")

    main(args)
