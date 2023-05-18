class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
from time import sleep
import time
from tqdm import tqdm
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset

from backbone import build_backbone
from matcher import build_matcher
from transformer import build_transformer
from detr import SetCriterion, DETR, PostProcess

from dreams_dataloader import dreams_dataset




def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr_drop', default=75, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=5, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=5, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=256, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=64, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=6, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


# Loading data, setting up GPU use, setting up variables for model training
batch_s = 3

print("Device used:")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def custom_collate(original_batch):
    label_list = []
    food_list = []
    for item in original_batch:
        food, label = item
        label_list.append(label)
        food_list.append(food)

    food_list = torch.stack(food_list)
    food_list = food_list.float()
    return food_list,label_list

dataset = dreams_dataset()
train_size = int(len(dataset)* 0.9)
val_size = int(len(dataset) - train_size)
dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size,val_size])
dataset_val = dataset_train

sampler_train = torch.utils.data.RandomSampler(dataset_train)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)


data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,collate_fn=custom_collate, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,drop_last=False, collate_fn=custom_collate, num_workers=args.num_workers)


EPOCHS = 1
NUM_CLASSES = 1
# Defining model
backbone = build_backbone(args)

transformer = build_transformer(args)
#print(backbone)
net = DETR(
    backbone,
    transformer,
    num_classes=NUM_CLASSES,
    num_queries=15,
    aux_loss=args.aux_loss,
)
net.to(device)

matcher = build_matcher(args)
weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
weight_dict['loss_giou'] = args.giou_loss_coef

losses = ['labels', 'boxes', 'cardinality']
criterion = SetCriterion(NUM_CLASSES, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
criterion.to(device)
postprocessors = {'bbox': PostProcess()}

param_dicts = [
        {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]


optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

correct = 0
total = 0

training_accuracy = []
training_loss = []
validation_accuracy = []
validation_loss = []


#=======================================================================#
# Running model training and validation loops. Model is fed with data, which
# split into 4 different frequencies. Accuracy and loss are displayed in the terminal output

for epoch in range(EPOCHS):  # loop over the dataset multiple times
    net.train()
    
    with tqdm(data_loader_train, unit="batch") as tepoch:
        running_acc = []
        running_loss = []
        labels_temp = []
        for i, data in enumerate(tepoch):
            #tepoch.set_description(f"{bcolors.WARNING} T Epoch {bcolors.ENDC} {epoch}")
            

            food, labels = data
            food = food.to(device)

            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            # Model training procedures
            optimizer.zero_grad()
            outputs = net(food)

            loss_dict = criterion(outputs, labels)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses.backward()
            optimizer.step()

            # Calculating metrics: loss and accuracy
            running_loss.append(losses.item())

            #correct = (predictions == labels).sum().item()
            #accuracy = correct / batch_s
            #running_acc.append(accuracy)
            
            # tqdm progress bar update
            if (i % 10 == 9) or (i == 0):
                tepoch.set_postfix(loss=sum(running_loss)/len(running_loss), accuracy=0)

    # Saving accuracy and loss values    
        

    net.eval()
    with tqdm(data_loader_val, unit="batch") as tepoch:
        running_acc = []
        running_loss = []
        for i, data in enumerate(tepoch):
        # get the inputs; data is a list of [inputs, labels]
            #tepoch.set_description(f"{bcolors.HEADER} V Epoch {bcolors.ENDC} {epoch}")
            # Loading the 4 different frequency bands and their respective label
            # Loading everything to device for GPU training
            food, labels = data
            food = food.to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in labels]

            # Model training procedures
            optimizer.zero_grad()
            outputs = net(food)
           # predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            loss_dict = criterion(outputs, labels)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # Calculating metrics: loss and accuracy
            running_loss.append(losses.item())
           # running_acc.append(accuracy)
            
            # tqdm progress bar update
            if (i % 10 == 9) or (i == 0):
                tepoch.set_postfix(loss=sum(running_loss)/len(running_loss), accuracy=0)
            sleep(0.1)
    
    # Saving accuracy and loss values 
    #validation_accuracy.append(100. * (sum(running_acc)/len(running_acc)))
    #validation_loss.append(loss.item())
print("=========================================LABELS======================================================")
print(labels)
print("=========================================OUTPUTS======================================================")
probas = outputs['pred_logits'].softmax(-1)[:, :, :-1]
keep = probas.max(-1).values > 0.7
#boxes_keep = outputs['pred_boxes'][:, keep]
#print(outputs['pred_logits'])
boxes_kept = []
for to_keep, box in zip(keep, outputs['pred_boxes']):
    boxes_kept.append(box[to_keep])


#print(probas)
#print(keep)
print(boxes_kept)
#print(probas.shape)
#print(keep.shape)
#print(outputs['pred_logits'])
#print(outputs['pred_logits'].shape)

# open file in write mode
with open('loss_real.txt', 'w') as fp:
    for item in running_loss:
        # write each item on a new line
        fp.write("%s\n" % item)
    print('Done')

#torch.save(net, '/home/marius/Documents/OneDrive/MSc/StartUP/Code/m1_stats_features.pt')
print('Finished Training')
