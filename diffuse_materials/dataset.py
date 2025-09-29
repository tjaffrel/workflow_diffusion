import functools
import random
from typing import Optional, Sequence, Tuple

from absl import logging
import tensorflow as tf
import numpy as np


class MOFDataset:

  def __init__(
      self, *,
      name: str,
      video_shape: Tuple[int, int, int, int],
      dataset_paths: Sequence[str],
      shuffle_buffer_size: int = 1000):
    self.video_shape = video_shape
    self.text_max_seq_len = video_shape[0]
    self.text_emb_dim = 128
    self._dataset_paths = dataset_paths
    self._shuffle_buffer_size = shuffle_buffer_size
    

  def _load_examples_from_files(
      self, *, shuffle_files: bool, readahead=False) -> tf.data.Dataset:
    dataset_cls = tf.data.TFRecordDataset
    sorted_input_files = sorted(
        sum(map(tf.io.gfile.glob, self._dataset_paths), []))[::-1]

    d = dataset_cls(sorted_input_files)
    return d

  def __len__(self) -> int:
        return len(self._dataset_paths)  

  def get_shuffled_repeated_dataset(
      self, host_batch_size: Optional[int]) -> tf.data.Dataset:
    d = self._load_examples_from_files(shuffle_files=True)
    d = d.repeat()
    d = d.map(
        lambda x: self.__getitem__(is_training=True, example=x),
        num_parallel_calls=tf.data.AUTOTUNE)
    d = d.apply(tf.data.experimental.ignore_errors())
    d = d.shuffle(buffer_size=self._shuffle_buffer_size)
    if host_batch_size is not None:
      d = d.batch(host_batch_size, drop_remainder=True)
    return d.prefetch(tf.data.AUTOTUNE)

  def __getitem__(self, is_training: bool, example):
    # Parse the TF example
    ex = tf.io.parse_single_example(example, {
        'frac_coords': tf.io.VarLenFeature(tf.float32),
        'atom_types': tf.io.VarLenFeature(tf.int64),
        'lengths': tf.io.VarLenFeature(tf.float32),
        'angles': tf.io.VarLenFeature(tf.float32),
        'formula': tf.io.VarLenFeature(tf.string),
    })

    out = {"indep_mask": tf.zeros([], dtype=tf.bool)}
    F, H, W, C = self.video_shape
    assert (H, W, C) == (1, 1, 4)

    NUM_ELEMENTS = 94
    MAX_NUM_ATOMS = self.text_max_seq_len - 2
    position = tf.reshape(tf.sparse.to_dense(ex['frac_coords']), [-1, 3])[:MAX_NUM_ATOMS]
    atoms = tf.sparse.to_dense(ex['atom_types'])[:MAX_NUM_ATOMS]
    graph_nodes = tf.one_hot(atoms - 1, self.text_emb_dim, dtype=tf.float32)
    atoms = tf.cast(atoms, tf.float32) / NUM_ELEMENTS

    # Order invariance
    #points = tf.random.shuffle(points)
    #graph_nodes = tf.random.shuffle(graph_nodes)

    # Lattice structure
    lengths = tf.reshape(tf.sparse.to_dense(ex['lengths']), [3])
    MIN_LEN, MAX_LEN = (0., 60.)
    lengths = (lengths - MIN_LEN) / (MAX_LEN - MIN_LEN)
    angles = tf.reshape(tf.sparse.to_dense(ex['angles']), [3])
    MIN_AGL, MAX_AGL = (0., 180.)
    angles = (angles - MIN_AGL) / (MAX_AGL - MIN_AGL)
    # Rotation invariance
    order = tf.random.shuffle([0, 1, 2])
    position = tf.gather(position, order, axis=-1)
    lengths = tf.gather(lengths, order)
    angles = tf.gather(angles, order)

    points = tf.concat([position, atoms[:, None]], axis=-1)
    points = tf.clip_by_value(points * 255, 0, 255)


    lattice = tf.stack([lengths, angles], axis=0)
    lattice = tf.concat([lattice, tf.zeros([2, 1])], axis=-1)
    lattice = tf.clip_by_value(lattice * 255, 0, 255)
    video = tf.concat([lattice, points], axis=0)
    if tf.shape(video)[0] < F:
        pad_len = F - tf.shape(video)[0]
        video = tf.concat([video, tf.zeros([pad_len, 4])], axis=0)
        frame_valid_mask = tf.concat([tf.ones([tf.shape(video)[0]], dtype=tf.bool),
                                      tf.zeros([pad_len], dtype=tf.bool)], axis=0)
    else:
        video = video[:F]
        frame_valid_mask = tf.ones([F], dtype=tf.bool)

    frame_valid_mask = tf.ones([F], dtype=tf.bool)

    video = video[:, None, None, :]
    video = tf.ensure_shape(video, [F, H, W, C])
    return video, graph_nodes
