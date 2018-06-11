import torch.nn as nn
import torch.nn.functional as F
import math
import random
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNeXt', 'resnext50', 'resnext101', 'resnext152']


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, base_width, cardinality, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		D = int(math.floor(planes * (base_width / 64)))
		C = cardinality
		self.conv1 = nn.Conv2d(inplanes, D*C, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(D*C)
		self.conv2 = nn.Conv2d(D*C, D*C, kernel_size=3, stride=stride,
							   padding=1, groups=C, bias=False)
		self.bn2 = nn.BatchNorm2d(D*C)
		self.conv3 = nn.Conv2d(D*C, planes * self.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

		global blockID
		self.blockID = blockID
		blockID += 1
		self.downsampling_ratio = 1.

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual

		if self.downsampling_ratio < 1:
			out = F.adaptive_avg_pool2d(out, int(round(out.size(2)*self.downsampling_ratio)))

		out = self.relu(out)

		return out


class ResNeXt(nn.Module):

	def __init__(self, block, layers, base_width=4, cardinality=32, num_classes=1000):
		self.cardinality = cardinality
		self.base_width = base_width
		self.inplanes = 64
		super(ResNeXt, self).__init__()

		global blockID
		blockID = 0

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
							   bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(512 * block.expansion, num_classes)

		self.blockID = blockID
		self.downsampling_ratio = 1.
		self.size_after_maxpool = None

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, self.base_width, self.cardinality, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, self.base_width, self.cardinality))

		return nn.Sequential(*layers)

	def stochastic_downsampling(self, blockID, ratio):
		block_chosen = blockID is None and random.randint(-1, self.blockID) or blockID
		downsampling_ratios = ratio is None and [0.5, 0.75] or [ratio, ratio]
		if self.blockID == block_chosen:
			self.downsampling_ratio = downsampling_ratios[random.randint(0,len(downsampling_ratios)-1)]
		else:
			self.downsampling_ratio = 1.
		for m in self.modules():
			if isinstance(m, Bottleneck):
				if m.blockID == block_chosen:
					m.downsampling_ratio = downsampling_ratios[random.randint(0,len(downsampling_ratios)-1)]
				else:
					m.downsampling_ratio = 1.

	def forward(self, x, blockID=None, ratio=None):
		self.stochastic_downsampling(blockID, ratio)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		if self.downsampling_ratio < 1:
			if self.size_after_maxpool is None:
				self.size_after_maxpool = self.maxpool(x).size(2)
			x = F.adaptive_max_pool2d(x, int(round(self.size_after_maxpool*self.downsampling_ratio)))
		else:
			x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def resnext50(pretrained=False, **kwargs):
	"""Constructs a ResNeXt-50 model.
	"""
	model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
	return model


def resnext101(pretrained=False, **kwargs):
	"""Constructs a ResNeXt-101 model.
	"""
	model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
	return model


def resnext152(pretrained=False, **kwargs):
	"""Constructs a ResNeXt-152 model.
	"""
	model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
	return model
