from vit import ViT
import os
import time
import torch
import visdom
import argparse
import torch.nn as nn
import torchvision.transforms as tfs
from torch.utils.data import DataLoader
from timm.models.layers import trunc_normal_
from torchvision.datasets.cifar import CIFAR10

def main():
    # 1. ** argparser **
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=50)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    parer.add_argument('--step_size', type=int, default=100)
    parer.add_argument('--root', type=str, default='./CIFAR10')
    parer.add_argument('--log_dir', type=str, default='./log')
    parer.add_argument('--name', type=str, default='vit_cifar10')
    parer.add_argument('--rank', type=int, default=0)
    ops = parer.parse_args()

    # 2. ** device **
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # 3. ** visdom **
    vis = visdom.Visdom(port=8097)

    # 4. ** dataset / dataloader **
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010)),
                                        ])
    train_set = CIFAR10(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR10(root=ops.root,
                       train=False,
                       download=True,
                       transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=ops.batch_size)

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=ops.batch_size)

    # 5. ** model **
    model = ViT().to(device)

    # 6. ** criterion **
    criterion = nn.CrossEntropyLoss()

    # 7. ** optimizer **
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ops.lr,
                                 weight_decay=5e-5)

    # 8. ** scheduler **
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=1e-5)

    # 9. ** logger **
    os.makedirs(ops.log_dir, exist_ok=True)

    # 10. ** training **
    print("training...")
    for epoch in range(ops.epoch):

        model.train()
        tic = time.time()
        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]
            # output, attn_mask = model(img, True)  # [N, 10]
            output = model(img)  # [N, 10]
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            if idx % ops.step_size == 0:
                vis.line(X=torch.ones((1, 1)) * idx + epoch * len(train_loader),
                         Y=torch.Tensor([loss]).unsqueeze(0),
                         update='append',
                         win='training_loss',
                         opts=dict(x_label='step',
                                   y_label='loss',
                                   title='loss',
                                   legend=['total_loss']))

                print('Epoch : {}\t'
                      'step : [{}/{}]\t'
                      'loss : {}\t'
                      'lr   : {}\t'
                      'time   {}\t'
                      .format(epoch,
                              idx, len(train_loader),
                              loss,
                              lr,
                              time.time() - tic))

        # save
        save_path = os.path.join(ops.log_dir, ops.name, 'saves')
        os.makedirs(save_path, exist_ok=True)

        checkpoint = {'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'scheduler_state_dict': scheduler.state_dict()}

        torch.save(checkpoint, os.path.join(save_path, ops.name + '.{}.pth.tar'.format(epoch)))

        # 10. ** test **
        print('Validation of epoch [{}]'.format(epoch))
        model.eval()
        correct = 0
        val_avg_loss = 0
        total = 0
        with torch.no_grad():

            for idx, (img, target) in enumerate(test_loader):
                model.eval()
                img = img.to(device)  # [N, 3, 32, 32]
                target = target.to(device)  # [N]
                output = model(img)  # [N, 10]
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                # first eval
                pred, idx_ = output.max(-1)
                correct += torch.eq(target, idx_).sum().item()
                total += target.size(0)
                val_avg_loss += loss.item()

        print('Epoch {} test : '.format(epoch))
        accuracy = correct / total
        print("accuracy : {:.4f}%".format(accuracy * 100.))

        val_avg_loss = val_avg_loss / len(test_loader)
        print("avg_loss : {:.4f}".format(val_avg_loss))

        if vis is not None:
            vis.line(X=torch.ones((1, 2)) * epoch,
                     Y=torch.Tensor([accuracy, val_avg_loss]).unsqueeze(0),
                     update='append',
                     win='test_loss',
                     opts=dict(x_label='epoch',
                               y_label='test_',
                               title='test_loss',
                               legend=['accuracy', 'avg_loss']))
        scheduler.step()


if __name__ == '__main__':
    main()

