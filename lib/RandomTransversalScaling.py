# -*- coding: utf-8 -*-

import torch
from torchvision.transforms import functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
import numbers

class RandomTransversalScaling(object):
	def __init__(self, p, plus_width=10, interpolation=InterpolationMode.BICUBIC):
		self.p = p
		self.plus_width = plus_width
		self.interpolation = interpolation

		if isinstance(interpolation, int):
			interpolation = _interpolation_modes_from_int(interpolation)
		self.interpolation = interpolation

	@staticmethod
	def get_params(width: list[float]) -> float:
		return float(torch.empty(1).uniform_(float(width[0]), float(width[1])).item())

	def _setup_width(self):
		x = self.plus_width
		if isinstance(x, numbers.Number):
			if x < 0:
				raise ValueError(f"If width is a single number, it must be positive.")
			x = [-x, x]
		x = [float(d) for d in x]
		return round(self.get_params(x))

	def _transversal_scaling(self, pil_img):
		img_width, img_height = pil_img.size
		add_width = self._setup_width()
		return F.resize(pil_img, [img_height, img_width + add_width], self.interpolation)

	def __call__(self, image):
		if torch.rand(1).item() < self.p:
			return self._transversal_scaling(image)
		return image
