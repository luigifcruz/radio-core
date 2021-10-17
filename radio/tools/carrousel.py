"""Defines a Carrousel module."""


class Carrousel:
    """
    The Carrousel class provides a ring buffer of elements.

    The operation of this class is similar to the collections.Queue.
    But unlike it, this class doesn't discard the element after its poped.
    This is more efficient, specially if using GPU memory.

    Attributes
    ----------
    items : arr
        array of items to be cycled through
    print_overflow : bool, optional
        print 'overflow' in stdout whenever an overflow happens (default is True)
    """

    def __init__(self, items, print_overflow: bool = True):
        self._items = items
        self._head = 0
        self._tail = 0
        self._overflows = 0
        self._occupancy = 0
        self._capacity = len(self._items)
        self._print_overflow = print_overflow

    @property
    def occupancy(self) -> int:
        """Returns the amount of items currently in use."""
        return self._occupancy

    @property
    def capacity(self) -> int:
        """Returns the total amount of items this instance can hold."""
        return self._capacity

    @property
    def is_empty(self) -> int:
        """Returns if there is any items in use."""
        return self.occupancy == 0

    @property
    def is_full(self) -> int:
        """Returns if all the items are the use."""
        return self.occupancy >= self.capacity

    @property
    def overflows(self) -> int:
        """Return the number of overflows since the instantiation."""
        return self._overflows

    @property
    def is_healthy(self) -> bool:
        """
        Returns if it's secure to read any item. Used to ensure buffer health.
        It's considered healthy if there is two or more items in use.
        """
        return self.occupancy >= 2

    def reset(self):
        """Reset class to the initial state."""
        self._head = 0
        self._tail = 0
        self._occupancy = 0

    def __str__(self):
        return self._items.__str__()

    def write(self):
        """Returns the reference of an item to be written into."""
        if self.is_full:
            if self._print_overflow:
                print('overflow')
            self.reset()

        _idx = self._tail
        self._occupancy += 1
        self._tail = (self._tail + 1) % self.capacity
        return self._items[_idx]

    def read(self):
        """Returns the reference of an item to be read."""
        if self.is_empty:
            raise ValueError('carrousel is empty')

        _idx = self._head
        self._occupancy -= 1
        self._head = (self._head + 1) % self.capacity
        return self._items[_idx]
