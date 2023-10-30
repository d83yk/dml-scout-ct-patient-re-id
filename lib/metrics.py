import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class AdaCos(nn.Module):
	def __init__(self, num_features, num_classes, m=0.50):
		super(AdaCos, self).__init__()
		self.num_features = num_features
		self.n_classes = num_classes
		self.s = math.sqrt(2) * math.log(num_classes - 1)
		self.m = m
		self.W = Parameter(torch.FloatTensor(num_classes, num_features))
		nn.init.xavier_uniform_(self.W)

	def forward(self, input, label=None):
		# normalize features
		x = F.normalize(input)
		# normalize weights
		W = F.normalize(self.W)
		# dot product
		logits = F.linear(x, W)
		if label is None:
			return logits
		# feature re-scale
		theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
		one_hot = torch.zeros_like(logits)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		with torch.no_grad():
			B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
			B_avg = torch.sum(B_avg) / input.size(0)
			# print(B_avg)
			theta_med = torch.median(theta[one_hot == 1])
			self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
		output = self.s * logits

		return output


class ArcFace(nn.Module):
	def __init__(self, num_features, num_classes, s=30.0, m=0.50):
		super(ArcFace, self).__init__()
		self.num_features = num_features
		self.n_classes = num_classes
		self.s = s
		self.m = m
		self.W = Parameter(torch.FloatTensor(num_classes, num_features))
		nn.init.xavier_uniform_(self.W)

	def forward(self, input, label=None):
		# normalize features
		x = F.normalize(input)
		# normalize weights
		W = F.normalize(self.W)
		# dot product
		logits = F.linear(x, W)
		if label is None:
			return logits
		# add margin
		theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
		target_logits = torch.cos(theta + self.m)
		one_hot = torch.zeros_like(logits)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		output = logits * (1 - one_hot) + target_logits * one_hot
		# feature re-scale
		output *= self.s

		return output


class SphereFace(nn.Module):
	def __init__(self, num_features, num_classes, s=30.0, m=1.35):
		super(SphereFace, self).__init__()
		self.num_features = num_features
		self.n_classes = num_classes
		self.s = s
		self.m = m
		self.W = Parameter(torch.FloatTensor(num_classes, num_features))
		nn.init.xavier_uniform_(self.W)

	def forward(self, input, label=None):
		# normalize features
		x = F.normalize(input)
		# normalize weights
		W = F.normalize(self.W)
		# dot product
		logits = F.linear(x, W)
		if label is None:
			return logits
		# add margin
		theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
		target_logits = torch.cos(self.m * theta)
		one_hot = torch.zeros_like(logits)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		output = logits * (1 - one_hot) + target_logits * one_hot
		# feature re-scale
		output *= self.s

		return output


class CosFace(nn.Module):
	def __init__(self, num_features, num_classes, s=30.0, m=0.35):
		super(CosFace, self).__init__()
		self.num_features = num_features
		self.n_classes = num_classes
		self.s = s
		self.m = m
		self.W = Parameter(torch.FloatTensor(num_classes, num_features))
		nn.init.xavier_uniform_(self.W)

	def forward(self, input, label=None):
		# normalize features
		x = F.normalize(input)
		# normalize weights
		W = F.normalize(self.W)
		# dot product
		logits = F.linear(x, W)
		if label is None:
			return logits
		# add margin
		target_logits = logits - self.m
		one_hot = torch.zeros_like(logits)
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		output = logits * (1 - one_hot) + target_logits * one_hot
		# feature re-scale
		output *= self.s

		return output

class ArcMarginProduct(nn.Module):
	r"""Implement of large margin arc distance: :
		Args:
			in_features: size of each input sample
			out_features: size of each output sample
			s: norm of input feature
			m: margin

			cos(theta + m)
		"""
	def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
		super(ArcMarginProduct, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

		self.easy_margin = easy_margin
		self.cos_m = math.cos(m)
		self.sin_m = math.sin(m)
		self.th = math.cos(math.pi - m)
		self.mm = math.sin(math.pi - m) * m

	def forward(self, input, label):
		# --------------------------- cos(theta) & phi(theta) ---------------------------
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
		phi = cosine * self.cos_m - sine * self.sin_m
		if self.easy_margin:
			phi = torch.where(cosine > 0, phi, cosine)
		else:
			phi = torch.where(cosine > self.th, phi, cosine - self.mm)
		# --------------------------- convert label to one-hot ---------------------------
		# one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
		one_hot = torch.zeros(cosine.size(), device='cuda')
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		# -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
		output *= self.s
		# print(output)

		return output


class AddMarginProduct(nn.Module):
	r"""Implement of large margin cosine distance: :
	Args:
		in_features: size of each input sample
		out_features: size of each output sample
		s: norm of input feature
		m: margin
		cos(theta) - m
	"""

	def __init__(self, in_features, out_features, s=30.0, m=0.40):
		super(AddMarginProduct, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.s = s
		self.m = m
		self.weight = Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform_(self.weight)

	def forward(self, input, label):
		# --------------------------- cos(theta) & phi(theta) ---------------------------
		cosine = F.linear(F.normalize(input), F.normalize(self.weight))
		phi = cosine - self.m
		# --------------------------- convert label to one-hot ---------------------------
		one_hot = torch.zeros(cosine.size(), device='cuda')
		# one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
		one_hot.scatter_(1, label.view(-1, 1).long(), 1)
		# -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
		output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
		output *= self.s
		# print(output)

		return output

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			   + 'in_features=' + str(self.in_features) \
			   + ', out_features=' + str(self.out_features) \
			   + ', s=' + str(self.s) \
			   + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
	r"""Implement of large margin cosine distance: :
	Args:
		in_features: size of each input sample
		out_features: size of each output sample
		m: margin
		cos(m*theta)
	"""
	def __init__(self, in_features, out_features, m=4):
		super(SphereProduct, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.m = m
		self.base = 1000.0
		self.gamma = 0.12
		self.power = 1
		self.LambdaMin = 5.0
		self.iter = 0
		self.weight = Parameter(torch.FloatTensor(out_features, in_features))
		nn.init.xavier_uniform(self.weight)

		# duplication formula
		self.mlambda = [
			lambda x: x ** 0,
			lambda x: x ** 1,
			lambda x: 2 * x ** 2 - 1,
			lambda x: 4 * x ** 3 - 3 * x,
			lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
			lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
		]

	def forward(self, input, label):
		# lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
		self.iter += 1
		self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

		# --------------------------- cos(theta) & phi(theta) ---------------------------
		cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
		cos_theta = cos_theta.clamp(-1, 1)
		cos_m_theta = self.mlambda[self.m](cos_theta)
		theta = cos_theta.data.acos()
		k = (self.m * theta / 3.14159265).floor()
		phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
		NormOfFeature = torch.norm(input, 2, 1)

		# --------------------------- convert label to one-hot ---------------------------
		one_hot = torch.zeros(cos_theta.size())
		one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
		one_hot.scatter_(1, label.view(-1, 1), 1)

		# --------------------------- Calculate output ---------------------------
		output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
		output *= NormOfFeature.view(-1, 1)

		return output

	def __repr__(self):
		return self.__class__.__name__ + '(' \
			   + 'in_features=' + str(self.in_features) \
			   + ', out_features=' + str(self.out_features) \
			   + ', m=' + str(self.m) + ')'
