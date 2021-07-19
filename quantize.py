from models.voc import mobilenetv2, mobilenetv3
from torch.hub import load_state_dict_from_url
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import argparse

def set_backend(backend):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend

def get_representative_dataset(valdir):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]))
    representative_dataset = torch.utils.data.Subset(val_dataset, list(range(100)))

    return representative_dataset

def quantize_model(model, backend, valdir=None):
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
    if valdir:
        representative_dataset = get_representative_dataset(valdir)
        dataloader = torch.utils.data.DataLoader(
                representative_dataset,
                batch_size=1,
                shuffle=True,
                )
        for (image, label) in tqdm(dataloader):
            model(image)
    else:
        _dummy_input_data = torch.rand(1, 3, 224, 224)
        model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

class QuantizableMobileNetV2(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(QuantizableMobileNetV2, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.net = mobilenetv2.mobilenetv2(pretrained, **kwargs)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x1, x2 = self.net.forward(x)
        x1 = self.dequant(x1)
        x2 = self.dequant(x2)
        return x1, x2

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, mobilenetv2.InvertedResidual) or isinstance(m, mobilenetv2.MobileNetV2):
                m.fuse_model()

def main(args):
    torch.manual_seed(563)
    model_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-c5e733a8.pth'
    set_backend(args.backend)

    if args.load:
        model = torch.jit.load(args.load)
    else:
        if args.model == 'mobilenetv2':
            model = QuantizableMobileNetV2(model_url)
        else:
            print("Selected model not available")
            exit(1)
        quantize_model(model, args.backend, args.valdir)

    model.eval()

    if args.save:
        torch.jit.save(torch.jit.script(model), args.save)

    start = time.time()
    img = torch.randn(1,3,224,224)
    with torch.no_grad():
        with trange(args.inference) as t:
            for _ in t:
                model.forward(img)
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
    args = parser.parse_args()
    main(args)
