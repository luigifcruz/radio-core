"""Defines a Carrousel module."""

from contextlib import contextmanager

from radiocore.tools import Buffer


class Carrousel:
    """
    The Carrousel class provides a ring buffer of elements.

    The operation of this class is similar to the collections.Queue.
    But unlike it, this class doesn't discard the element after its poped.
    This is more efficient, specially if using GPU memory.

    This class doesn't support multiple producers.

    Attributes
    ----------
    items : arr
        array of items to be cycled through
    print_overflow : bool, optional
        print 'overflow' in stdout whenever some happens (default is True)
    """

    def __init__(self, items, print_overflow: bool = True):
        """Initialize the Carrousel class."""
        self._items = items
        self._head = 0
        self._tail = 0
        self._overflow = 0
        self._occupancy = 0
        self._capacity = len(self._items)
        self._print_overflow = print_overflow

    @property
    def occupancy(self) -> int:
        """Return the amount of items currently in use."""
        return self._occupancy

    @property
    def capacity(self) -> int:
        """Return the total amount of items this instance can hold."""
        return self._capacity

    @property
    def is_empty(self) -> int:
        """Return if there is any items in use."""
        return self.occupancy == 0

    @property
    def is_full(self) -> int:
        """Return if all the items are the use."""
        return self.occupancy >= self.capacity

    @property
    def overflow(self) -> int:
        """Return the number of overflows since the instantiation."""
        return self._overflow

    @property
    def is_healthy(self) -> bool:
        """
        Return if it's secure to read any item. Used to ensure buffer health.

        It's considered healthy if there is two or more items in use.
        """
        return self.occupancy >= 1

    def reset(self):
        """Reset class to the initial state."""
        self._head = 0
        self._tail = 0
        self._occupancy = 0

    def __str__(self):
        """Return the data inside the buffer in string format."""
        return self._items.__str__()

    @contextmanager
    def enqueue(self):
        """Return the reference of an item to be written into."""
        if self.is_full:
            self._overflow += 1
            self._occupancy -= 1
            self._head = (self._head + 1) % self.capacity

            if self._print_overflow:
                print("overflow")

        try:
            _item = self._items[self._tail]
            if isinstance(_item, Buffer):
                with _item.consume() as _buf:
                    yield _buf
            else:
                yield _item
        finally:
            self._occupancy += 1
            self._tail = (self._tail + 1) % self.capacity

    @contextmanager
    def dequeue(self):
        """Return the reference of an item to be read."""
        if self.is_empty:
            raise ValueError("carrousel is empty")

        try:
            _item = self._items[self._head]
            if isinstance(_item, Buffer):
                with _item.consume() as _buf:
                    yield _buf
            else:
                yield _item
        finally:
            self._occupancy -= 1
            self._head = (self._head + 1) % self.capacity
