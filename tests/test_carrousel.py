"""Carrousel test."""

from radio import Carrousel


def test_carrousel():
    """Test carrousel function."""
    carsl = Carrousel([[0], [0], [0]])

    assert carsl.occupancy == 0
    assert carsl.capacity == 3
    assert not carsl.is_full
    assert carsl.is_empty

    buf = carsl.write()
    buf[0] = 1

    assert carsl.occupancy == 1
    assert carsl.capacity == 3
    assert not carsl.is_full
    assert not carsl.is_empty

    buf = carsl.write()
    buf[0] = 2

    assert carsl.occupancy == 2
    assert carsl.capacity == 3
    assert not carsl.is_full
    assert not carsl.is_empty

    buf = carsl.write()
    buf[0] = 3

    assert carsl.occupancy == 3
    assert carsl.capacity == 3
    assert carsl.is_full
    assert not carsl.is_empty

    print(carsl)

    buf = carsl.read()
    assert buf[0] == 1

    assert carsl.occupancy == 2
    assert carsl.capacity == 3
    assert not carsl.is_full
    assert not carsl.is_empty

    buf = carsl.read()
    assert buf[0] == 2

    assert carsl.occupancy == 1
    assert carsl.capacity == 3
    assert not carsl.is_full
    assert not carsl.is_empty

    buf = carsl.read()
    assert buf[0] == 3

    assert carsl.occupancy == 0
    assert carsl.capacity == 3
    assert not carsl.is_full
    assert carsl.is_empty
