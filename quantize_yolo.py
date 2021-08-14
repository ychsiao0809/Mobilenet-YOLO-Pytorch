import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from models.voc import mobilenetv2, mobilenetv3, mbv2_yolo
from torch.hub import load_state_dict_from_url
from tqdm import tqdm, trange
from inference import load_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import argparse
import yaml
import folder2lmdb

def set_backend(backend):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend

def quantize_model(model, backend, test_loader=None):
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.default_observer,
                weight=torch.quantization.default_weight_observer)

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    if test_loader:
        for idx, (image, label) in tqdm(enumerate(test_loader)):
            if idx == 20:
                break
            model(image)
    else:
        _dummy_input_data = torch.rand(1, 3, 224, 224)
        model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

class QuantizableYoloWithoutLoss(nn.Module):
    def __init__(self, config, checkpoint):
        super(QuantizableYoloWithoutLoss, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.yolo = mbv2_yolo.YoloWithoutLoss(config)
        self.yolo = load_model(self.yolo, checkpoint, 'cpu')
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        out0, out1 = self.yolo(x)
        out0 = self.dequant(out0)
        out1 = self.dequant(out1)
        return (out0, out1)

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, mobilenetv2.InvertedResidual) or isinstance(m, mobilenetv2.MobileNetV2):
                m.fuse_model()

class QuantizableYolo(nn.Module):
    def __init__(self, config, quantized_yolowithoutloss):
        super(QuantizableYolo, self).__init__()
        self.yolo = quantized_yolowithoutloss
        self.yololoss = mbv2_yolo.YoloLoss(config)

    def forward(self, x, targets=None):
        h, w = x.size(2), x.size(3)
        out0, out1 = self.yolo.forward(x)
        return self.yololoss.forward(out0, out1, targets, h, w)

    @property
    def yolo_losses(self):
        return self.yololoss.yolo_losses


def main(args):
    torch.manual_seed(563)
    set_backend(args.backend)

    yaml_file = 'models/voc/config.yaml'
    with open(yaml_file, 'r') as f:
        config = yaml.load(f)

    with open(f'data/voc_data.yaml', 'r') as f:
        dataset_path = yaml.load(f)

    image_folder = folder2lmdb.ImageFolderLMDB
    test_dataset = image_folder(
            db_path=f'{dataset_path["test_dataset_path"]["lmdb"]}',
            transform_size=[[config["img_w"],config["img_h"]]],
            phase='test',batch_size=1
            )
    test_loader = torch.utils.data.DataLoader(
            test_dataset, 1, shuffle=False,
            num_workers=4, pin_memory=True,collate_fn=test_dataset.collate_fn)

    if args.load:
        model = torch.jit.load(args.load)
    else:
        if args.model == 'mobilenetv2':
            model = QuantizableYoloWithoutLoss(config=config, checkpoint=args.checkpoint)
        else:
            print("Selected model not available")
            exit(1)
        quantize_model(model, args.backend, test_loader)

    model.eval()

    if args.save:
        torch.jit.save(torch.jit.script(model), args.save)

    yolo = QuantizableYolo(config, model)
    yolo.eval()
    yolo.yololoss.yolo_losses[0].val_conf = 0.01
    yolo.yololoss.yolo_losses[1].val_conf = 0.01

    start = time.time()
    img = torch.randn(1,3,224,224)
    with torch.no_grad():
        with trange(args.inference) as t:
            for _ in t:
                yolo.forward(img)
    end = time.time()
    print(end - start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save", default=None, help="Save model", type=str)
    parser.add_argument("-l", "--load", default=None, help="Load model from checkpoint", type=str)
    parser.add_argument("-i", "--inference", default=20, help="Number of inference", type=int)
    parser.add_argument("-m", "--model", default="mobilenetv2", help="Choose model type", type=str)
    parser.add_argument("-b", "--backend", default="qnnpack", help="Select backend for PyTorch", type=str)
    parser.add_argument('--valdir', dest="valdir", required=False, type=str)
    parser.add_argument('-c', '--checkpoint', default='checkpoint/model_best.pth.tar', type=str, metavar='PATH',
                        help='path to load checkpoint (default: checkpoint/model_best.pth.tar)')
    args = parser.parse_args()
    main(args)
