# -*- coding:utf-8 -*-
import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib import ticker as mticker
from matplotlib import rcParams

from BasicTools import unit_convert

rcParams['axes.axisbelow'] = False


class ERBScale(mscale.ScaleBase):

    name = 'erb'

    def get_transform(self):
        """return a mtransforms instance, where axis converting functions
            (forward_f,invert_f) are defined
        """
        return self.ERBTransform()

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """limit the bounds of axis
        """
        MIN_VALUE = 1e-20  # avoid NAN because of log10
        return max(vmin, MIN_VALUE), vmax

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(mticker.AutoLocator())
        axis.set_minor_locator(mticker.AutoLocator())

    class ERBTransform(mtransforms.Transform):
        """
        """
        # value members that required
        input_dims = 1  # dimension of input (source axis)
        output_dims = 1  # dimension of output (target axis)
        # True if this transform is separable in the x- and y- dimensions.
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return unit_convert.hz2erbscale(a)

        def inverted(self):
            return ERBScale.InvertERBTransform()

    class InvertERBTransform(mtransforms.Transform):
        """
        """
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
            return unit_convert.erbscale2hz(a)

        def inverted(self):
            return ERBScale.ERBTransform()


mscale.register_scale(ERBScale)


if __name__ == "__main__":
    x = np.arange(100, 2000, 100)
    y = x
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y)
    ax.set_yscale('erb')
    ax.set_xlabel('Frequency(Hz)')
    ax.set_ylabel('ERB scale')
    fig.savefig('../../tmp.png')
