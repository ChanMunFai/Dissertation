"""
Frechet Video Distance (FVD) is a metric for the quality of video generation models. It is
inspired by the FID (Frechet Inception Distance) used for images, but
uses a different embedding to be better suitable for videos.

Code is obtained from (Apache 2.0 License)
https://github.com/google-research/google-research/blob/master/frechet_video_distance/frechet_video_distance.py
"""

import argparse
from copy import deepcopy
from typing import Any, Tuple

import numpy as np
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

class FrechetVideoDistance: 
    """
    """
    def __init__(self):
        pass

    def preprocess(videos: tf.Tensor, target_resolution: Tuple[int, int]) -> Any:
        """
        Run some preprocessing on the videos for I3D model.
        :param videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
        preprocessed. We don't care about the specific dtype of the videos, it can
        be anything that tf.image.resize_bilinear accepts. Values are expected to
        be in the range 0-255.
        :param target_resolution: (width, height): target video resolution
        :return: videos: <float32>[batch_size, num_frames, height, width, depth]
        """

        videos_shape = videos.shape.as_list()
        all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
        resized_videos = tf.image.resize(all_frames, size=target_resolution)
        target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
        output_videos = tf.reshape(resized_videos, target_shape)
        scaled_videos = 2.0 * tf.cast(output_videos, tf.float32) / 255.0 - 1
        return scaled_videos

    def _is_in_graph(tensor_name: tf.Tensor) -> bool:
        """
        Check whether a given tensor does exists in the graph.
        """
        try:
            tf.get_default_graph().get_tensor_by_name(tensor_name)
        except KeyError:
            return False
        return True


    def create_id3_embedding(videos: tf.Tensor, batch_size: int) -> tf.Tensor:
        """
        Embed the given videos using the Inflated 3D Convolution network.
        Downloads the graph of the I3D from tf.hub and adds it to the graph on the
        first call.
        :param videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3]. Expected range is [-1, 1].
        :param batch_size: batch size
        :return: <float32>[batch_size, embedding_size]. embedding_size depends on the model used.
        :raises ValueError: when a provided embedding_layer is not supported.
        """

        module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"

        # Making sure that we import the graph separately for
        # each different input video tensor.
        module_name = "fvd_kinetics-400_id3_module_" + videos.name.replace(":", "_")

        assert_ops = [
            tf.Assert(tf.reduce_max(videos) <= 1.001, ["max value in frame is > 1", videos]),
            tf.Assert(tf.reduce_min(videos) >= -1.001, ["min value in frame is < -1", videos]),
            tf.assert_equal(tf.shape(videos)[0], batch_size, ["invalid frame batch size: ", tf.shape(videos)], summarize=6),
        ]
        with tf.control_dependencies(assert_ops):
            videos = tf.identity(videos)

        module_scope = "%s_apply_default/" % module_name

        # To check whether the module has already been loaded into the graph, we look
        # for a given tensor name. If this tensor name exists, we assume the function
        # has been called before and the graph was imported. Otherwise we import it.
        # Note: in theory, the tensor could exist, but have wrong shapes.
        # This will happen if create_id3_embedding is called with a frames_placehoder
        # of wrong size/batch size, because even though that will throw a tf.Assert
        # on graph-execution time, it will insert the tensor (with wrong shape) into
        # the graph. This is why we need the following assert.
        video_batch_size = int(videos.shape[0])
        assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
        tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
        if not _is_in_graph(tensor_name):
            i3d_model = hub.Module(module_spec, name=module_name)
            i3d_model(videos)

        # gets the kinetics-i3d-400-logits layer
        tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
        tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
        return tensor


    def calculate_fvd(real_activations: tf.Tensor, generated_activations: tf.Tensor) -> tf.Tensor:
        """
        Return a list of ops that compute metrics as funcs of activations.
        :param real_activations: <float32>[num_samples, embedding_size]
        :param generated_activations: <float32>[num_samples, embedding_size]
        :return: FVD score
        """
        return tf.contrib.gan.eval.frechet_classifier_distance_from_activations(real_activations, generated_activations)


