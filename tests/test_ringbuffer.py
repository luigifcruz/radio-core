"""Ring Buffer test."""

import numpy as np

from radiocore import RingBuffer


def test_buffer():
    """Test ring buffer function."""
    a = RingBuffer(8, dtype=np.float32)
    assert a.occupancy == 0
    assert a.capacity == 8
    assert a.vacancy == 8
    assert np.allclose(a.data, [0., 0., 0., 0., 0., 0., 0., 0.])

    a.append([1, 2, 3, 4])
    assert a.occupancy == 4
    assert a.capacity == 8
    assert a.vacancy == 4
    assert np.allclose(a.data, [1., 2., 3., 4., 0., 0., 0., 0.])
    print(a, a.occupancy)

    a.append([5, 6, 7, 8])
    assert a.occupancy == 8
    assert a.capacity == 8
    assert a.vacancy == 0
    assert np.allclose(a.data, [1., 2., 3., 4., 5., 6., 7., 8.])
    print(a, a.occupancy)

    b = np.zeros(4, dtype=np.float32)
    a.popleft(b)
    assert a.occupancy == 4
    assert a.capacity == 8
    assert a.vacancy == 4
    assert np.allclose(a.data, [1., 2., 3., 4., 5., 6., 7., 8.])
    assert np.allclose(b, [1., 2., 3., 4.])
    print(a, b, a.occupancy)

    a.append([1, 1, 1, 1])
    assert a.occupancy == 8
    assert a.capacity == 8
    assert a.vacancy == 0
    assert np.allclose(a.data, [1., 1., 1., 1., 5., 6., 7., 8.])
    print(a, a.occupancy)

    a.append([1, 1, 1, 1])
    assert a.occupancy == 8
    assert a.capacity == 8
    assert a.vacancy == 0
    assert np.allclose(a.data, [1., 1., 1., 1., 1., 1., 1., 1.])
    print(a, a.occupancy)
