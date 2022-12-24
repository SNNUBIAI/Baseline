import torch
import torch.nn as nn
from sklearn.linear_model import Lasso
import numpy as np

from baseline.masker import Masker
from baseline.utils import thresholding

class Autoencoder(nn.Module):
	def __init__(self, dim=28546, n_components=64):
		super(Autoencoder, self).__init__()
		self.fc1 = nn.Linear(in_features=dim, out_features=n_components)
		self.fc2 = nn.Linear(in_features=n_components, out_features=dim)

	def forward(self, x):
		encode = self.fc1(x)
		decode = self.fc2(encode)
		return encode, decode

class AE:
	def __init__(self,
				 img,
				 mask_img=None,
				 n_components=64,
				 dim=28546,
				 lr=0.001,
				 epochs=5,
				 batch_size=1,
				 device="cuda",
				 alpha=0.001):
		self.ae = Autoencoder(dim=dim, n_components=n_components)
		self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)
		self.mse_loss = nn.MSELoss()
		self.epochs = epochs
		self.batch_size = batch_size
		self.device = device
		self.alpha = alpha
		self.lasso = Lasso(alpha)
		if mask_img:
			self.masker = Masker(mask_path=mask_img)
		else:
			self.masker = None

		self.components_ = None
		self.ae.to(self.device)

		self.data = torch.tensor(img, dtype=torch.float).to(self.device)

	def fit(self, epochs=5):
		self.ae.train()
		self.epochs = epochs
		for epoch in range(self.epochs):
			total_loss = 0
			cnt = 0
			for i in range(0, self.data.shape[0], self.batch_size):
				if i + self.batch_size <= self.data.shape[0]:
					x_data = self.data[i:i+self.batch_size, :]
				else:
					x_data = self.data[i:, :]
				_, y_data = self.ae(x_data)
				loss = self.mse_loss(x_data, y_data)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				total_loss += loss.item()
				cnt += 1

			total_loss = total_loss / cnt
			print("Epoch {}/{} : loss={:.4f}".format(epoch+1, epochs, total_loss))

		# print("Extracting the sources......")
		# sources = self.encode()
		# print("Generating FBNs via Lasso......")
		# self.lasso.fit(sources, self.data.detach().cpu().numpy())
		# self.components_ = self.lasso.coef_.T

	def get_components_(self):
		if self.components_ is not None:
			return self.components_

		print("Extracting the sources......")
		sources = self.encode(self.data)
		print("Generating FBNs via Lasso......")
		self.lasso.fit(sources, self.data.detach().cpu().numpy())
		self.components_ = self.lasso.coef_.T
		return self.components_

	@torch.no_grad()
	def predict(self, img):
		data = torch.tensor(img, dtype=torch.float).to(self.device)
		print("Extracting the sources......")
		sources = self.encode(data)
		print("Generating FBNs via Lasso......")
		self.lasso.fit(sources, data.detach().cpu().numpy())
		self.components_ = self.lasso.coef_.T
		return self.components_

	def save_model(self, path):
		torch.save(self.ae.state_dict(), path)

	def load_model(self, path):
		self.ae.load_state_dict(torch.load(path))

	@torch.no_grad()
	def encode(self, data):
		self.ae.eval()
		encode_ls = []
		for i in range(0, data.shape[0], self.batch_size):
			if i + self.batch_size <= data.shape[0]:
				x_data = data[i:i + self.batch_size, :]
			else:
				x_data = data[i:, :]
			encode, _ = self.ae(x_data)
			encode_ls.append(encode.detach().cpu().numpy())
		sources = np.concatenate(encode_ls, axis=0)
		return sources
