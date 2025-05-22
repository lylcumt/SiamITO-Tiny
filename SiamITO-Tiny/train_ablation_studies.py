import torchvision.models.alexnet
import visdom
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset import Pair
from tracking import val
from model import Backbone, SiameseITO, BCEWeightLoss

if __name__ == '__main__':
    train_path = './infrared_small_object_data/train'
    val_path = './infrared_small_object_data/val'
    resume = False
    backbone_type = 'alexnet'
    epoch = 10000
    batch_size = 32
    best_score = 0

    transform = transforms.ToTensor()

    dataset = Pair(train_path, transforms=transform, rand_choice=True, gary=False,
                   search_size=127, template_size=65, r_pos=6)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if resume:
        model_path = './train_model/best.pkl'
        net = torch.load(model_path)
    elif backbone_type == 'alexnet':
        backbone = torchvision.models.alexnet().features
        net = SiameseITO(embedding_net=backbone, upscale=True, upscale_size=65)
    elif backbone_type == 'resnet':
        backbone = torchvision.models.resnet18()
        net = SiameseITO(embedding_net=backbone, upscale=True, upscale_size=65)
    else:
        backbone = Backbone()
        net = SiameseITO(embedding_net=backbone, upscale=True, upscale_size=65)

    net.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.5, 0.999))
    criterion = BCEWeightLoss()

    losses = 0
    is_vis = False  # 是否进行可视化，如果没有visdom可以将其设置为false
    vis_fre = 5
    if is_vis:
        vis = visdom.Visdom()
        viswin1 = vis.line(np.array([0.]), np.array([0.]),
                           opts=dict(title="Loss/epoch", xlabel="%d*epoch" % vis_fre, ylabel="Loss"))

    for e in range(epoch):
        if (e + 1) % 5000 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.2
        for index, (template, search, label, weight) in enumerate(dataloader):
            template = template.cuda()
            search = search.cuda()
            label = label.cuda()
            weight = weight.cuda()
            out = net(template, search)

            loss = criterion(out, label, weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.data

        if (e+1) % vis_fre == 0:
            mean_loss = losses / (len(dataloader)*vis_fre)
            print('epoch:', e+1, ' mean_loss:', mean_loss.cpu().data.numpy() / batch_size)
            losses = 0

            if is_vis:
                vis.line(np.array([mean_loss.cpu().data/batch_size]), np.array([e+1]), win=viswin1, update='append')

        if (e + 1) % 500 == 0:
            net.eval()
            total_score = val(val_path, net)
            print(f'epoch {e+1} val result: {total_score}')

            if total_score > best_score:
                if backbone_type == 'alexnet':
                    torch.save(net, './train_model/best_alexnet.pkl')
                elif backbone_type == 'resnet':
                    torch.save(net, './train_model/best_resnet18.pkl')
                torch.save(net, './train_model/best.pkl')

            if backbone_type == 'alexnet':
                torch.save(net, './train_model/SiamITO_alexnet_last.pkl')
            elif backbone_type == 'resnet':
                torch.save(net, './train_model/SiamITO_resnet18_last.pkl')
            torch.save(net, './train_model/SiamITO_last.pkl')

            net.train()
