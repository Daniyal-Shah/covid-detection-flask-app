


import tensorflow as tf
import splitfolders


# Split with a ratio.
splitfolders.ratio('./data', output="./dataset",
    seed=1337, ratio=(.7, .3), group_prefix=None, move=False) # default values

