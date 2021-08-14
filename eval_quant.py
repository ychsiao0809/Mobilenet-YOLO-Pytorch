import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from models.voc import mbv2_yolo
from torch.hub import load_state_dict_from_url
from inference import load_model
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import yaml
import folder2lmdb
import argparse
from inference import load_model
from train import test
from quantize_yolo import QuantizableYolo, set_backend

def main(args):
    set_backend(args.backend)
    torch.multiprocessing.set_sharing_strategy('file_system')
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

    model = torch.jit.load(args.load)
    yolo = QuantizableYolo(config, model)
    yolo.eval()
    yolo.yololoss.yolo_losses[0].val_conf = 0.01
    yolo.yololoss.yolo_losses[1].val_conf = 0.01

    best_acc = test(test_loader, yolo, None, 0, config)
    print(best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", default=None, help="Load model from checkpoint", type=str)
    parser.add_argument("-b", "--backend", default="fbgemm", help="Select backend for PyTorch", type=str)
    args = parser.parse_args()
    main(args)
