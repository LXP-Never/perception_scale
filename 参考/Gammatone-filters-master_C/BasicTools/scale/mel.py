# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib import ticker as mticker
from matplotlib import rcParams

from .. import unit_convert

rcParams['axes.axisbelow'] = False


class MelScale(mscale.ScaleBase):

    name = 'mel'

    def get_transform(self):
        """return a mtransforms instance, where axis converting functions
            (forward_f,invert_f) are defined
        """
        return self.MelTransform()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """limit the bounds of axis
        """
        MIN_VALUE = 1e-20  # freq > 0
        return max(vmin, MIN_VALUE), vmax

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.AutoLocator())
        axis.set_minor_locator(mticker.AutoLocator())

    class MelTransform(mtransforms.Transform):
        """
        """
        # value members that required
        input_dims = 1  # dimension of input (source axis)
        output_dims = 1  # dimension of output (target axis)
        # True if this transform is separable in the x- and y- dimensions.
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, a):
            return unit_convert.hz2mel(a)

        def inverted(self):
            return MelScale.InvertMelTransform()

    class InvertMelTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return unit_convert.mel2hz(a)

        def inverted(self):
            return MelScale.MelTransform()


mscale.register_scale(MelScale)


if __name__ == "__main__":
    x = np.arange(100, 2000, 100)
    y = x
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    # ax.plot(x, y)
    ax.imshow(np.random.rand(20, 20), extent=[100, 3e3, 100, 3e3])
    ax.set_yscale('mel')
    ax.set_xlabel('Frequency(hz)')
    ax.set_ylabel('Mel scale')
    fig.savefig('../../tmp.png')
