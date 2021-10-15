from contextlib import contextmanager
import importlib
import threading


class Buffer:


    def __init__(self, size: int, lock: bool = True, dtype = 'complex64', cuda: bool = False):
        self.__mtx = threading.Lock()
        self.__lock = lock
        self.__cuda = cuda
        self.__dtype = dtype
        self.__size = size

        self.__load_modules()
        self.__load_buffer()


    def __load_modules(self):
        if self.__cuda:
            self.__sg = importlib.import_module('cusignal')
        else:
            self.__np = importlib.import_module('numpy')


    def __load_buffer(self):
        if self.__cuda:
            self.__buffer = self.__sg.get_shared_mem(self.__size, dtype = self.__dtype)
        else:
            self.__buffer = self.__np.zeros(self.__size, dtype = self.__dtype)


    @property
    def dtype(self):
        return self.__buffer.dtype


    @property
    def is_cuda(self) -> bool:
        return self.__cuda


    @property
    def size(self) -> int:
        return self.__size


    @property
    def is_locked(self) -> bool:
        return self.__mtx.locked()


    @contextmanager
    def consume(self):
        try:
            if self.__lock:
                self.__mtx.acquire()
            yield self.__buffer
        finally:
            if self.__lock:
                self.__mtx.release()
