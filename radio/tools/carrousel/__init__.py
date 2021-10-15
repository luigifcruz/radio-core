from contextlib import contextmanager
import importlib
import threading


class Carrousel:


    def __init__(self, items, print_overflow: bool = True):
        self.__items = items
        self.__head = 0
        self.__tail = 0
        self.__overflows = 0
        self.__occupancy = 0
        self.__capacity = len(self.__items)
        self.__print_overflow = print_overflow


    @property
    def occupancy(self) -> int:
        return self.__occupancy


    @property
    def capacity(self) -> int:
        return self.__capacity


    @property
    def is_empty(self) -> int:
        return self.occupancy == 0


    @property
    def is_full(self) -> int:
        return self.occupancy >= self.capacity


    @property
    def overflows(self) -> int:
        return self.__overflows


    @property
    def is_healthy(self) -> bool:
        return self.occupancy >= 2


    def __str__(self):
        return self.__items.__str__()


    def write(self):
        if self.is_full:
            if self.__print_overflow:
                print('overflow')
            self.__overflows += 1
            self.__occupancy -= 1

        _idx = self.__tail
        self.__occupancy += 1
        self.__tail = (self.__tail + 1) % self.capacity
        return self.__items[_idx]


    def read(self):
        if self.is_empty:
            raise ValueError('carrousel is empty')

        _idx = self.__head
        self.__occupancy -= 1
        self.__head = (self.__head + 1) % self.capacity
        return self.__items[_idx]
