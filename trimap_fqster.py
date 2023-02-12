import datetime
import time
from typing import Mapping
import numpy as np
from absl import logging
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pynndescent
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from trimap import tempered_log, get_distance_fn, rejection_sample, sample_random_triplets, transform, trimap_loss, update_embedding_dbd, trimap_metrics

# logging = logging.getlogging("__info__")

_DIM_PCA = 100
_INIT_SCALE = 0.01
_INIT_MOMENTUM = 0.5
_FINAL_MOMENTUM = 0.8
_SWITCH_ITER = 250
_MIN_GAIN = 0.01
_INCREASE_GAIN = 0.2
_DAMP_GAIN = 0.8
_DISPLAY_ITER = 100

def sample_knn_triplets(key, neighbors, n_inliers, n_outliers, max_index):
  """Sample nearest neighbors triplets based on the neighbors.

  Args:
    key: Random key.
    neighbors: Nearest neighbors indices for each point.
    self.n_inliers: Number of inliers.
    self.n_outliers: Number of outliers.

  Returns:
    triplets: Sampled triplets.
  """
  n_points = neighbors.shape[0]
  anchors = jnp.tile(
      jnp.arange(n_points).reshape([-1, 1]),
      [1, n_inliers * n_outliers]).reshape([-1, 1])
  inliers = jnp.tile(neighbors[:, 1:n_inliers + 1],
                     [1, n_outliers]).reshape([-1, 1])
  outliers = rejection_sample(key, (n_points, n_inliers * n_outliers), max_index,
                              neighbors).reshape([-1, 1])
  triplets = jnp.concatenate((anchors, inliers, outliers), 1)
  return triplets

def sliced_distances(
    indices1,
    indices2,
    test_inputs,
    train_inputs,
    distance_fn):
  """Applies distance_fn in smaller slices to avoid memory blow-ups.

  Args:
    indices1: First array of indices.
    indices2: Second array of indices.
    inputs: 2-D array of inputs.
    distance_fn: Distance function that applies row-wise.

  Returns:
    Pairwise distances between the row indices in indices1 and indices2.
  """
  slice_size = test_inputs.shape[0]
  distances = []
  num_slices = int(np.ceil(len(indices1) / slice_size))
  for slice_id in range(num_slices):
    start = slice_id * slice_size
    end = (slice_id + 1) * slice_size
    distances.append(
        distance_fn(test_inputs[indices1[start:end]], train_inputs[indices2[start:end]]))
  return jnp.concatenate(distances)





class TriMap():
  def __init__(self, 
               key,
               n_dims=2,
               n_inliers=10,
               n_outliers=5,
               n_random=3,
               weight_temp=0.5,
               distance='euclidean',
               lr=0.1,
               n_iters=400,
               init_embedding='pca',
               apply_pca=True,
               verbose=True) -> None:

    self.key = key
    self.n_dims = n_dims
    self.n_inliers = n_inliers
    self.n_outliers = n_outliers
    self.n_random = n_random
    self.weight_temp = weight_temp
    self.distance = distance
    self.lr = lr
    self.n_iters = n_iters
    self.init_embedding = init_embedding
    self.apply_pca = apply_pca
    self.verbose = verbose
    self.train_triplets = None
    self.train_weights = None
    self.test_triplets = None
    self.test_weights = None
    self.distance_fn = get_distance_fn(self.distance)


  def fit(self, inputs):
      
    if self.verbose:
      print("Sampling train triplets")
    self.train_triplets, self.train_weights = self.generate_triplets(inputs, train_time=True)
    if self.verbose:
      print("Done")
    self.train_embeddings, self.components = transform(
      self.key,
      inputs,
      self.n_dims,
      self.n_inliers,
      self.n_outliers,
      self.n_random,
      self.weight_temp,
      self.distance,
      self.lr,
      self.n_iters,
      self.init_embedding,
      self.apply_pca,
      triplets=self.train_triplets,
      weights = self.train_weights,
      verbose=self.verbose
    )
    self.train = inputs
    self.n_train = len(self.train)
         

  def transform(self, inputs):
    if self.verbose:
      t = time.time()
    n_points = len(inputs)
    test_triplets, test_weights = self.generate_triplets(inputs, train_time=False)
    embeddings = inputs.dot(self.components.T)
    n_triplets = float(test_triplets.shape[0])
    lr = self.lr * n_points / n_triplets
    if self.verbose:
      logging.info('running TriMap using DBD')
    vel = jnp.zeros_like(embeddings, dtype=jnp.float32)
    gain = jnp.ones_like(embeddings, dtype=jnp.float32)
    trimap_grad = jax.jit(jax.grad(trimap_loss))
    for itr in range(self.n_iters):
      gamma = _FINAL_MOMENTUM if itr > _SWITCH_ITER else _INIT_MOMENTUM
      grad = trimap_grad(embeddings + gamma * vel, test_triplets, test_weights)

      # update the embedding
      embedding, vel, gain = update_embedding_dbd(embeddings, grad, vel, gain, lr,
                                                  itr)
      if self.verbose:
        if (itr + 1) % _DISPLAY_ITER == 0:
          loss, n_violated = trimap_metrics(embeddings, test_triplets, test_weights)
          logging.info(
              'Iteration: %4d / %4d, Loss: %3.3f, Violated triplets: %0.4f',
              itr + 1, self.n_iters, loss, n_violated / n_triplets * 100.0)
    if self.verbose:
      elapsed = str(datetime.timedelta(seconds=time.time() - t))
      logging.info('Elapsed time: %s', elapsed)
    return embeddings


  def find_triplet_weights(self,
                         inputs,
                         triplets,
                         neighbors,
                         distance_fn,
                         sig,
                         distances=None,
                         train_time=True):
    """Calculates the weights for the sampled nearest neighbors triplets.

    Args:
      inputs: Input points.
      triplets: Nearest neighbor triplets.
      neighbors: Nearest neighbors.
      distance_fn: Distance function.
      sig: Scaling factor for the distances
      distances: Nearest neighbor distances.

    Returns:
      weights: Triplet weights.
    """
    n_points, self.n_inliers = neighbors.shape
    if distances is None:
      anchs = jnp.tile(jnp.arange(n_points).reshape([-1, 1]),
                      [1, self.n_inliers]).flatten()
      inliers = neighbors.flatten()
      distances = sliced_distances(anchs, inliers, inputs, inputs, distance_fn)**2
      p_sim = -distances / (sig[anchs] * sig[inliers])
    else:
      p_sim = -distances.flatten()
    self.n_outliers = triplets.shape[0] // (n_points * self.n_inliers)
    p_sim = jnp.tile(p_sim.reshape([n_points, self.n_inliers]),
                    [1, self.n_outliers]).flatten()
    if train_time:
      out_distances = sliced_distances(triplets[:, 0], triplets[:, 2], inputs, inputs, distance_fn)**2
    else: 
      out_distances = sliced_distances(triplets[:, 0], triplets[:, 2], inputs, self.train, distance_fn)**2
    p_out = -out_distances / (sig[triplets[:, 0]] * sig[triplets[:, 2]])
    weights = p_sim - p_out
    return weights

  def find_scaled_neighbors(self, inputs, neighbors, distance_fn, train_time):
    """Calculates the scaled neighbors and their similarities.

    Args:
      inputs: Input examples.
      neighbors: Nearest neighbors
      distance_fn: Distance function.

    Returns:
      Scaled distances and neighbors, and the scale parameter.
    """
    n_points, n_neighbors = neighbors.shape
    anchors = jnp.tile(jnp.arange(n_points).reshape([-1, 1]),
                      [1, n_neighbors]).flatten()
    hits = neighbors.flatten()
    if train_time:
      distances = sliced_distances(anchors, hits, inputs, inputs, distance_fn)**2
    else:
      distances = sliced_distances(anchors, hits, inputs, self.train, distance_fn)**2
    distances = distances.reshape([n_points, -1])
    sig = jnp.maximum(jnp.mean(jnp.sqrt(distances[:, 3:6]), axis=1), 1e-10)
    scaled_distances = distances / (sig.reshape([-1, 1]) * sig[neighbors])
    sort_indices = jnp.argsort(scaled_distances, 1)
    scaled_distances = jnp.take_along_axis(scaled_distances, sort_indices, 1)
    sorted_neighbors = jnp.take_along_axis(neighbors, sort_indices, 1)
    return scaled_distances, sorted_neighbors, sig
  
  def sample_random_test_triplets(self, key, inputs, n_random, distance_fn, sig):
    n_test = inputs.shape[0]
    anchors = jnp.tile(jnp.arange(n_test).reshape([-1, 1]),
                      [1, n_random]).reshape([-1, 1])
    _, use_key = random.split(key)
    pairs = random.randint(use_key, shape=(n_test * n_random, 2), minval=0, maxval=self.n_train)
    triplets = jnp.concatenate((anchors, pairs), 1)
    anc = triplets[:, 0]
    sim = triplets[:, 1]
    out = triplets[:, 2]
    p_sim = -(sliced_distances(anc, sim, inputs, self.train, distance_fn)**2) / (
        sig[anc] * sig[sim])
    p_out = -(sliced_distances(anc, out, inputs, self.train, distance_fn)**2) / (
        sig[anc] * sig[out])
    flip = p_sim < p_out
    weights = p_sim - p_out
    pairs = jnp.where(
        jnp.tile(flip.reshape([-1, 1]), [1, 2]), jnp.fliplr(pairs), pairs)
    triplets = jnp.concatenate((anchors, pairs), 1)
    return triplets, weights

  def generate_triplets(self, inputs, train_time=True):
    
    n_points = inputs.shape[0]
    n_extra = min(self.n_inliers + 50, n_points)
    if self.verbose:
      print("Generqte index")
    if train_time:
      self.index = pynndescent.NNDescent(inputs, metric=self.distance)
      self.index.prepare()
    neighbors = self.index.query(inputs, k=n_extra)[0]
    neighbors = np.concatenate((np.arange(n_points).reshape([-1, 1]), neighbors),1)
    if self.verbose:
      print('found nearest neighbors')
    knn_distances, neighbors, sig = self.find_scaled_neighbors(inputs, neighbors, self.distance_fn, train_time)
    neighbors = neighbors[:, :self.n_inliers + 1]
    knn_distances = knn_distances[:, :self.n_inliers + 1]
    key, use_key = random.split(self.key)
    if train_time:
      triplets = sample_knn_triplets(use_key, neighbors, self.n_inliers, self.n_outliers, len(neighbors))
      weights = self.find_triplet_weights(inputs, triplets, neighbors[:, 1:self.n_inliers + 1], self.distance_fn, sig,
        distances=knn_distances[:, 1:self.n_inliers + 1])
    else:
      triplets = sample_knn_triplets(use_key, neighbors, self.n_inliers, self.n_outliers, len(self.train))
      weights = self.find_triplet_weights(inputs, triplets, neighbors[:, 1:self.n_inliers + 1], self.distance_fn, sig, distances=knn_distances[:, 1:self.n_inliers + 1], train_time=False)

    flip = weights < 0
    anchors, pairs = triplets[:, 0].reshape([-1, 1]), triplets[:, 1:]
    pairs = jnp.where(
        jnp.tile(flip.reshape([-1, 1]), [1, 2]), jnp.fliplr(pairs), pairs)
    triplets = jnp.concatenate((anchors, pairs), 1)

    if self.n_random > 0:
      key, use_key = random.split(key)
      if train_time:
        rand_triplets, rand_weights = sample_random_triplets(
            use_key, inputs, self.n_random, self.distance_fn, sig)
      else:
        rand_triplets, rand_weights = self.sample_random_test_triplets(
            use_key, inputs, self.n_random, self.distance_fn, sig)

      triplets = jnp.concatenate((triplets, rand_triplets), 0)
      weights = jnp.concatenate((weights, 0.1 * rand_weights))

    if train_time:
      self.min_weight = jnp.min(weights)
      weights -= jnp.min(weights)
      
    else:
      weights -= jnp.minimum(self.min_weight, weights)
    weights = tempered_log(1. + weights, self.weight_temp)
    return triplets, weights
      


if __name__ == "__main__":
  import jax.numpy as jnp
  import polars as pl     
  data = "/Users/evrardgarcelon/Desktop/data/"
  X_train = pl.read_csv(data + "input_training.csv")
  X_test = pl.read_csv(data + "input_test.csv")
  from trimap_fqster import *
  import jax.random as random
  key = random.PRNGKey(1)
  trimap_reduction = TriMap(
    key=key,
    verbose=True)
  X_train = X_train.fill_null(0).sample(1000).to_numpy()
  print(X_train.shape)
  triplets, weights = trimap_reduction.generate_triplets(X_train)
  train_embeddings = trimap_reduction.fit(X_train)
  X_test = X_test.fill_null(0).sample(100).to_numpy()
  test_embeddings = trimap_reduction.transform(X_test)
  print(test_embeddings)

