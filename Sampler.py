import numpy as np


class Sampler(object):
	def __init__(self,mu=0.,sigma=0.15,n_distributions=10):
		self.sigma = sigma
		self.mu = mu
		self.n_distributions=n_distributions

	def get_sample(self, dimension, batch_size, n_distributions):
		pass


class MultivariateGaussianSampler(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		current_dist_states_indices = np.random.randint(0, n_distributions - 1, batch_size)
		mean_vec = np.linspace(-self.mu,self.mu,n_distributions)
		cov_mat = np.eye(n_distributions) * self.sigma  # np.random.randint(1, 5, n_distributions)  # this is diagonal beacuse we want iid

		result_vec = np.zeros((batch_size, embedding_dim))
		# create multimodal matrix
		matrix_sample = np.random.multivariate_normal(mean_vec, cov_mat, size=batch_size * embedding_dim)
		# matrix_sample from the multimodal matrix
		for i in range(batch_size):
			tmp = matrix_sample.reshape(embedding_dim, n_distributions, batch_size)
			result_vec[i] = tmp[:, current_dist_states_indices[i], i]
		return np.asarray(result_vec, dtype=np.float32)

class MultivariateGaussianTruncatedSampler(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		current_dist_states_indices = np.random.randint(0, n_distributions - 1, batch_size)
		mean_vec = np.linspace(-self.mu,self.mu,n_distributions)
		cov_mat = np.eye(n_distributions) * self.sigma  # np.random.randint(1, 5, n_distributions)  # this is diagonal beacuse we want iid

		result_vec = np.zeros((batch_size, embedding_dim))
		# create multimodal matrix
		matrix_sample = np.random.multivariate_normal(mean_vec, cov_mat, size=batch_size * embedding_dim)
		# matrix_sample from the multimodal matrix
		for i in range(batch_size):
			tmp = matrix_sample.reshape(embedding_dim, n_distributions, batch_size)
			result_vec[i] = tmp[:, current_dist_states_indices[i], i]
		return np.asarray(result_vec, dtype=np.float32)

class UniformSample(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		return np.random.uniform(-1, 1, size=(batch_size, embedding_dim)).astype(np.float32)


class GaussianSample(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		return np.random.normal(loc=self.mu, scale=self.sigma, size=(batch_size, embedding_dim)).astype(np.float32)

class TruncatedGaussianSample(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		import scipy.stats
		lower = -1
		upper = 1
		mu = self.mu
		sigma = self.sigma

		samples = scipy.stats.truncnorm.rvs(lower,upper,loc=mu,scale=sigma,size=(batch_size, embedding_dim))
		return  samples

class MultiModalUniformSample(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		means = np.linspace(-0.1,0.1,n_distributions)
		rand = np.random.randint(0, n_distributions - 1, batch_size)
		current_dist_states_indices= means[rand]
		result_vec = np.zeros((batch_size, embedding_dim))
		for i in range(batch_size):
			result_vec[i] = np.random.uniform(-1 + current_dist_states_indices[i], 1 + current_dist_states_indices[i], size=embedding_dim)
		return np.asarray(result_vec, dtype=np.float32)


class MultimodelGaussianTF(Sampler):
	def get_sample(self, batch_size, embedding_dim, n_distributions):
		import tensorflow as tf
		tfd = tf.contrib.distributions
		mu = np.arange(n_distributions, dtype=np.float32)
		sigma = np.ones(n_distributions, dtype=np.float32)
		bimix_gauss = tfd.Mixture(cat=tfd.Categorical(probs=np.ones(n_distributions, dtype=np.float32) / n_distributions),
			components=[tfd.Normal(loc=m, scale=s) for m, s in zip(mu, sigma)])
		return bimix_gauss.sample(embedding_dim * batch_size)


if __name__ == '__main__':
	bimix_gauss = MultimodelGaussianTF()
	import tensorflow as tf
	g= MultivariateGaussianSampler(mu=0.1,sigma=0.12)

	gg = g.get_sample(1000,10,10)
	print(np.max(gg))
	print(np.min(gg))
	print(np.mean(gg))
	# plt.plot(gg)

	# sess = tf.Session()
	# with sess.as_default():
	# 	d = bimix_gauss.get_sample(10, 5, 10).eval()
	# test = MultivariateGaussianSampler()
	# a = test.get_sample(10, 5, 3)
	# test_uni = UniformSample()
	# test_mul_uni=MultiModalUniformSample()
	# b = test_uni.get_sample(10, 5, 10)
	# c = test_mul_uni.get_sample(10, 5, 10)
	# print(d)
	# print(b)
	# print(c
	# plt.plot(a)
	# plt.show()
	# plt.plot(d)
	# plt.show()
	# print(gg)
	# plt.plot(gg)
	# plt.show()
	# plt.plot(b)
	# plt.show()
	# plt.plot(c)
	# plt.show()
	# plt.close()
	import seaborn as sns
	import pandas as pd

	# sns.pairplot(pd.DataFrame(a))
	# sns.jointplot(a[:, 0], a[:, 1], kind="hex", color="#4CB391", ylim=(-10, 10), xlim=(-14, 14))
	# plt.show()
