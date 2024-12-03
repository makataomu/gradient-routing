# %%
"""
Pre-routing architectures taken from:
https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

from copy import deepcopy
from typing import Callable

import torch as t
import torch.nn.functional as F
from line_profiler import profile
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    @profile
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RoutedResNet(nn.Module):
    def __init__(
        self,
        num_blocks,
        split_at_block,
        num_classes: int,
        use_split_encoder: bool,
        use_shared_decoder: bool,
    ):
        super().__init__()
        self.in_planes = 64

        self.use_split_encoder = use_split_encoder
        self.use_shared_decoder = use_shared_decoder

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        block = BasicBlock
        layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(4)
        self.flatten = nn.Flatten()
        layers = (
            layer1
            + layer2
            + layer3
            + layer4
            + [self.avgpool, self.flatten, self.linear]
        )
        encoder_layers = layers[:split_at_block]
        decoder_layers = layers[split_at_block:]

        in_channels = decoder_layers[0].conv1.in_channels
        encoder_layers += [
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        ]  # WARNING: this might break if splitting between block types

        self.encoder = nn.Sequential(*encoder_layers)

        decoder_types = {
            "top_decoder": False,
            "bot_decoder": False,
            "top_certificate": True,
            "bot_certificate": True,
        }

        self.decoders = nn.ModuleDict(
            {
                name: self._create_decoder(decoder_layers, halve_first_layer_input)
                for name, halve_first_layer_input in decoder_types.items()
            }
        )

        if self.use_shared_decoder:
            self.decoders["bot_decoder"] = self.decoders["top_decoder"]

    def _create_decoder(self, layers, halve_first_layer_input: bool):
        input_channels = layers[0].conv1.in_channels
        if halve_first_layer_input:
            resize = nn.Conv2d(
                input_channels // 2, input_channels, kernel_size=1, bias=False
            )
        else:
            resize = nn.Conv2d(
                input_channels, input_channels, kernel_size=1, bias=False
            )
        return nn.Sequential(resize, *deepcopy(layers))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return layers

    @profile
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        encoding = self.encoder(out)
        encoding_top, encoding_bot = encoding.split(encoding.shape[1] // 2, dim=1)
        encoding_top_detached = encoding_top.detach()
        encoding_bot_detached = encoding_bot.detach()

        if self.use_split_encoder:
            encoding_top_route = t.cat((encoding_top, encoding_bot_detached), dim=1)
            encoding_bot_route = t.cat((encoding_top_detached, encoding_bot), dim=1)
        else:
            encoding_top_route = encoding
            encoding_bot_route = encoding

        routed_top = self.decoders["top_decoder"](encoding_top_route)
        routed_bot = self.decoders["bot_decoder"](encoding_bot_route)
        top_certificate = self.decoders["top_certificate"](encoding_top_detached)
        bot_certificate = self.decoders["bot_certificate"](encoding_bot_detached)

        outputs = {
            "routed_top": routed_top,
            "routed_bot": routed_bot,
            "top_certificate": top_certificate,
            "bot_certificate": bot_certificate,
        }
        info = {
            "encoding": encoding,
        }

        return outputs, info

    def get_losses(
        self, x: t.Tensor, labels: t.Tensor, mask_fn: Callable, reduction: str
    ):
        out, info = self(x)
        is_top = mask_fn(labels)
        losses = {
            "loss_top": F.cross_entropy(
                out["routed_top"][is_top], labels[is_top], reduction=reduction
            ),  # NOTE: this would not respect class imbalance, if it existed
            "loss_bot": F.cross_entropy(
                out["routed_bot"][~is_top], labels[~is_top], reduction=reduction
            ),
            "loss_top_cert": F.cross_entropy(
                out["top_certificate"], labels, reduction=reduction
            ),
            "loss_bot_cert": F.cross_entropy(
                out["bot_certificate"], labels, reduction=reduction
            ),
            "l1": info["encoding"].abs().sum(dim=1).mean(),
            "l2": t.sqrt((info["encoding"] ** 2).sum(dim=1).mean()),
        }
        return losses


class PreloadedDataLoader:
    def __init__(self, dataset, batch_size, shuffle, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

        all_data = next(
            iter(t.utils.data.DataLoader(self.dataset, batch_size=len(self.dataset)))
        )
        self.data = [
            data.to(device) for data in all_data[:2]
        ]  # <-- IGNORING COARSE LABELS HERE

        self.indices = t.arange(len(self.dataset))

    def __iter__(self):
        self.current = 0
        if self.shuffle:
            self.indices = t.randperm(len(self.dataset))
        return self

    @profile
    def __next__(self):
        if self.current >= len(self.dataset):
            raise StopIteration

        start_index = self.current
        end_index = min(start_index + self.batch_size, len(self.dataset))
        batch_indices = self.indices[start_index:end_index]

        self.current += self.batch_size
        return [d[batch_indices] for d in self.data]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


@t.inference_mode()
def evaluate_accuracy(model, testloader):
    model.eval()
    num_correct = 0
    total = 0
    for x, labels, _ in testloader:
        out = model(x)
        is_correct = out.argmax(dim=1) == labels
        num_correct += is_correct.sum().item()
        total += len(labels)
    model.train()
    return num_correct / total


if __name__ == "__main__":
    pass
