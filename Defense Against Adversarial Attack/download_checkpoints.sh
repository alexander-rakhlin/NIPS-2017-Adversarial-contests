#!/bin/bash
#
# Scripts which download checkpoints
#

wget https://www.dropbox.com/s/d6j1tghzaw497hc/inception_v3_adv_from_tf.tar.gz
tar -xvzf inception_v3_adv_from_tf.tar.gz
rm inception_v3_adv_from_tf.tar.gz

wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
