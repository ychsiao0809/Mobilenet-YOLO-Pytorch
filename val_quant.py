from models.voc import mobilenetv2

from torch.hub import load_state_dict_from_url
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse

def set_backend(backend):
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend

def get_val_dataset(valdir):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
        ]))

    return val_dataset

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(1280, 1000)

    def forward(self, x):
        return self.classifier(x)

def classifier_init(pretrained):
    model = Classifier()
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = load_state_dict_from_url(pretrained,progress=True)

        for k1, v1 in checkpoint.items() :
            for k2, v2 in model_dict.items() :
                if k1 == k2 :
                    model_dict[k2]=v1

        model.load_state_dict(model_dict)

    return model

def main(args):
    model_url = 'https://raw.githubusercontent.com/d-li14/mobilenetv2.pytorch/master/pretrained/mobilenetv2-c5e733a8.pth'
    set_backend(args.backend)

    model = torch.jit.load(args.load)
    model.eval()
    classifier = classifier_init(model_url)
    classifier.eval()

    val_dataset = get_val_dataset(args.valdir)
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers
        )

    correct = 0
    with torch.no_grad():
        with tqdm(dataloader) as t:
            for (image, label) in t:
                pred = classifier(F.adaptive_avg_pool2d(model(image)[1], (1,1)).view(image.size(0), -1)).argmax(1)
                correct += (pred == label).sum().item()
                t.set_postfix(correct=correct)

    print(correct)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", help="Load model from checkpoint", type=str, required=True)
    parser.add_argument("-b", "--backend", default="fbgemm", help="Select backend for PyTorch", type=str)
    parser.add_argument("--valdir", dest="valdir", required=True, type=str)
    parser.add_argument("--num_workers", dest="num_workers", default=8, type=int)
    args = parser.parse_args()
    main(args)
