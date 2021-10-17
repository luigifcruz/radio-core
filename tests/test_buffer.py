"""Buffer test."""

import numpy as np

from radiocore import Buffer


def test_buffer():
    """Test buffer function."""
    bfo = Buffer(8, dtype='float')

    with bfo.consume() as buf:
        print(buf)
        assert np.allclose(buf, [0., 0., 0., 0., 0., 0., 0., 0.])

    with bfo.consume() as buf:
        buf[0] = 1
        buf[1] = 1

    with bfo.consume() as buf:
        print(buf)
        assert np.allclose(buf, [1., 1., 0., 0., 0., 0., 0., 0.])

    with bfo.consume() as buf:
        buf[2] = 2
        buf[3] = 2

    with bfo.consume() as buf:
        print(buf)
        assert np.allclose(buf, [1., 1., 2., 2., 0., 0., 0., 0.])
