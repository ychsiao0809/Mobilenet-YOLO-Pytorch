from models.voc import mbv2_yolo
from torch.hub import load_state_dict_from_url
from inference import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import yaml
import folder2lmdb
from inference import load_model
from train import test

def main():
    with open(f'models/voc/config.yaml', 'r') as f:
        config = yaml.load(f)

    with open(f'data/voc_data.yaml', 'r') as f:
        dataset_path = yaml.load(f)

    image_folder = folder2lmdb.ImageFolderLMDB
    test_dataset = image_folder(
            db_path=f'{dataset_path["test_dataset_path"]["lmdb"]}',
            transform_size=[[config["img_w"],config["img_h"]]],
            phase='test',batch_size=64
            )
    test_loader = torch.utils.data.DataLoader(
            test_dataset, 64, shuffle=False,
            num_workers=4, pin_memory=True,collate_fn=test_dataset.collate_fn)

    yolo = mbv2_yolo.yolo(config=config)
    yolo = load_model(yolo, './checkpoint/model_best.pth.tar')
    yolo.eval()
    yolo.yolo_losses[0].val_conf = 0.01 
    yolo.yolo_losses[1].val_conf = 0.01 
    yolo.cuda()

    best_acc = test(test_loader, yolo, None, 0, config)
    print(best_acc)

if __name__ == '__main__':
    main()
