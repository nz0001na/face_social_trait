# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import imp
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "../utils.py"))


###############################################################################
###############################################################################
###############################################################################


if __name__ == "__main__":
    # Load an image.
    # Need to download examples images first.
    # See script in images directory.
    image = utils.load_image(
        os.path.join(base_dir, "../images", "N2C_0332.jpg"), 224)

    # Code snippet.
    plt.imshow(image/255)
    plt.axis('off')
    plt.savefig("images/zn/input.png")

    import innvestigate
    import innvestigate.utils
    import keras.applications.vgg16 as vgg16

    # Get model
    model, preprocess = vgg16.VGG16(), vgg16.preprocess_input
    # Strip softmax layer
    model = innvestigate.utils.model_wo_softmax(model)

    # Create analyzer: map.keys()
    # analyzers = {
    #     # Utility.
    #     "input": Input,
    #     "random": Random,
    #
    #     # Gradient based
    #     "gradient": Gradient,
    #     "gradient.baseline": BaselineGradient,
    #     "input_t_gradient": InputTimesGradient,
    #     "deconvnet": Deconvnet,
    #     "guided_backprop": GuidedBackprop,
    #     "integrated_gradients": IntegratedGradients,
    #     "smoothgrad": SmoothGrad,
    #
    #     # Relevance based
    #     "lrp": LRP,
    #     "lrp.z": LRPZ,
    #     "lrp.z_IB": LRPZIgnoreBias,
    #
    #     "lrp.epsilon": LRPEpsilon,
    #     "lrp.epsilon_IB": LRPEpsilonIgnoreBias,
    #
    #     "lrp.w_square": LRPWSquare,
    #     "lrp.flat": LRPFlat,
    #
    #     "lrp.alpha_beta": LRPAlphaBeta,
    #
    #     "lrp.alpha_2_beta_1": LRPAlpha2Beta1,
    #     "lrp.alpha_2_beta_1_IB": LRPAlpha2Beta1IgnoreBias,
    #     "lrp.alpha_1_beta_0": LRPAlpha1Beta0,
    #     "lrp.alpha_1_beta_0_IB": LRPAlpha1Beta0IgnoreBias,
    #     "lrp.z_plus": LRPZPlus,
    #     "lrp.z_plus_fast": LRPZPlusFast,
    #
    #     "lrp.sequential_preset_a": LRPSequentialPresetA,
    #     "lrp.sequential_preset_b": LRPSequentialPresetB,
    #     "lrp.sequential_preset_a_flat": LRPSequentialPresetAFlat,
    #     "lrp.sequential_preset_b_flat": LRPSequentialPresetBFlat,
    #     "lrp.sequential_preset_b_flat_until_idx": LRPSequentialPresetBFlatUntilIdx,
    #
    #
    #     # Deep Taylor
    #     "deep_taylor": DeepTaylor,
    #     "deep_taylor.bounded": BoundedDeepTaylor,
    #
    #     # DeepLIFT
    #     #"deep_lift": DeepLIFT,
    #     "deep_lift.wrapper": DeepLIFTWrapper,
    #
    #     # Pattern based
    #     "pattern.net": PatternNet,
    #     "pattern.attribution": PatternAttribution,
    # }
    analyzer = innvestigate.create_analyzer("lrp.epsilon", model)

    # Add batch axis and preprocess
    x = preprocess(image[None])
    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(x)

    # Aggregate along color channels and normalize to [-1, 1]
    a = a.sum(axis=np.argmax(np.asarray(a.shape) == 3))
    a /= np.max(np.abs(a))
    # Plot
    plt.imshow(a[0], cmap="seismic", clim=(-1, 1))
    plt.axis('off')
    plt.savefig("images/zn/output.png")
