import ipdb
import pathlib
import jax.numpy as jnp
import numpy as np
import resource
import einops

low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
import tensorflow as tf
import tensorflow_datasets as tfds


def load(config):
    possible_dirs = config.data.possible_dirs
    while len(possible_dirs) > 0:
        possible_dir = pathlib.Path(possible_dirs.pop(0))
        try:
            builder = tfds.builder('isaac3d', data_dir=possible_dir)
            builder.download_and_prepare(download_dir=possible_dir / 'downloads')
            break
        except PermissionError as e:
            print(e)
    train_set = builder.as_dataset(split=tfds.Split.TRAIN)
    info = builder.info

    dataset_info = {}
    dataset_info['source_names'] = [
        k.replace('label_', '') for k in info.features.keys() if k.startswith('label_')
    ]
    dataset_info['num_sources'] = len(dataset_info['source_names'])
    label_names = [
        k for k in info.features.keys() if k.startswith('label_')
    ]
    dataset_info['num_train'] = info.splits['train'].num_examples
    sources_max = tf.cast(tf.stack([info.features[k].num_classes - 1 for k in label_names], axis=0), tf.float32)
    sources_min = tf.zeros_like(sources_max)

    def prepare_data(data):
        ret = {}
        x = data['image']
        x = tf.image.resize(x, [64, 64], method='bicubic', antialias=True)
        x = einops.rearrange(x, 'h w c -> c h w')
        x = tf.cast(x, tf.float32) / 255.
        x = tf.clip_by_value(x, 0., 1.)
        x = x * 2 - 1  # [-1, 1]
        ret['x'] = x

        sources = tf.cast(tf.stack([data[k] for k in label_names], axis=0), tf.float32)
        sources = (sources - sources_min) / (sources_max - sources_min) * 2 - 1  # [-1, 1]
        ret['s'] = sources
        return ret



    def filter_fn(data):
        cond_0 = tf.math.equal(data['label_object_shape'], 2)
        cond_1 = tf.math.equal(data['label_robot_x'], 0)
        cond_2 = tf.math.equal(data['label_robot_y'], 0)
        cond_4 = tf.math.equal(data['label_object_scale'], 3)
        cond_7 = tf.math.equal(data['label_object_color'], 3)
        # cond_0 = tf.math.equal(s[0], 0)
        # cond_1 = tf.math.equal(s[1], -1)
        # cond_2 = tf.math.equal(s[2], 0)
        # cond_4 = tf.math.equal(s[4], 1)
        # cond_7 = tf.math.equal(s[7], 1)
        return tf.math.logical_and(tf.math.logical_and(tf.math.logical_and(tf.math.logical_and(cond_0, cond_1), cond_2), cond_4), cond_7)

    if config.data.reduced:
        reduced_sources = [3, 5, 6, 8]
        dataset_info['num_sources'] = len(reduced_sources)
        dataset_info['source_names'] = [dataset_info['source_names'][i] for i in reduced_sources]

        def prepare_data_reduced(data):
            ret = {}
            x = data['image']
            x = tf.image.resize(x, [64, 64], method='bicubic', antialias=True)
            x = einops.rearrange(x, 'h w c -> c h w')
            x = tf.cast(x, tf.float32) / 255.
            x = tf.clip_by_value(x, 0., 1.)
            x = x * 2 - 1  # [-1, 1]
            ret['x'] = x

            sources = tf.cast(tf.stack([data[label_names[i]] for i in reduced_sources], axis=0), tf.float32)
            ret['s'] = sources
            return ret

        train_set = train_set.filter(filter_fn)
        val_set = train_set
        train_set = train_set.shuffle(train_set.cardinality().numpy(), seed=config.data.seed, reshuffle_each_iteration=True) \
            .repeat() \
            .map(prepare_data_reduced, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(config.data.batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        val_set = val_set.map(prepare_data_reduced, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(config.data.batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        val_set = train_set.take(config.data.num_val)
        train_set = train_set.shuffle(config.data.buffer_size, seed=config.data.seed, reshuffle_each_iteration=True) \
            .repeat() \
            .map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(config.data.batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        val_set = val_set.map(prepare_data, num_parallel_calls=tf.data.AUTOTUNE) \
            .batch(config.data.batch_size) \
            .prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset_info, tfds.as_numpy(train_set), tfds.as_numpy(val_set)
