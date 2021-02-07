import os
import torch
import torchvision
import argparse

from model import load_model, save_model
from modules import NT_Xent
from modules.transformations import TransformsSimCLR
from utils import mask_correlated_samples, post_config_hook
from pprint import pprint
from utils.yaml_config_hook import yaml_config_hook

import numpy as np
from data_helper import SimclrUnlabeledDataset
from helper import convert_map_to_lane_map, convert_map_to_road_map, collate_fn, draw_box


def train(args, train_loader, model, criterion, optimizer):
    loss_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):

        optimizer.zero_grad()
        x_i = x_i.to(args.device)
        x_j = x_j.to(args.device)

        # positive pair, with encoding
        h_i, z_i = model(x_i)
        h_j, z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        torch.cuda.empty_cache()
        #writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
        loss_epoch += loss.item()
        args.global_step += 1

    return loss_epoch




config = yaml_config_hook("./config/config.yaml")
args = argparse.Namespace(**config)


args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.out_dir = "logs/simclrPretext"

if not os.path.exists("logs/simclrPretext"):
    os.makedirs("logs/simclrPretext")
    
args.batch_size = 2
args.epochs = 100
args.epoch_num = 100
args.resnet = "resnet18"
args.dataset = "road"
args.model_path = "logs/simclrPretext"
#pprint(vars(args))


image_folder = 'data'
annotation_csv = 'data/annotation.csv'

unlabeled_scene_index = np.arange(106)



root = "./datasets"

train_sampler = None

if args.dataset == "STL10":
    train_dataset = torchvision.datasets.STL10(
        root, split="unlabeled", download=True, transform=TransformsSimCLR()
    )
elif args.dataset == "CIFAR10":
    train_dataset = torchvision.datasets.CIFAR10(
        root, download=True, transform=TransformsSimCLR()
    )
elif args.dataset == "road":
    train_dataset = SimclrUnlabeledDataset(image_folder=image_folder, 
      scene_index=unlabeled_scene_index, first_dim='sample', transform=TransformsSimCLR())
else:
    raise NotImplementedError

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=(train_sampler is None),
    drop_last=True,
    num_workers=args.workers,
    sampler=train_sampler,
)


model, optimizer, scheduler = load_model(args, train_loader)

mask = mask_correlated_samples(args)
criterion = NT_Xent(args.batch_size, args.temperature, mask, args.device)


args.global_step = 0
args.current_epoch = 0
for epoch in range(args.start_epoch, args.epochs):
    lr = optimizer.param_groups[0]['lr']
    loss_epoch = train(args, train_loader, model, criterion, optimizer)

    if scheduler:
        scheduler.step()

    if epoch % 1 == 0:
        save_model(args, model, optimizer)

    #writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
    #writer.add_scalar("Misc/learning_rate", lr, epoch)
    print(
        f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t lr: {round(lr, 5)}"
    )
    args.current_epoch += 1

## end training
save_model(args, model, optimizer)

