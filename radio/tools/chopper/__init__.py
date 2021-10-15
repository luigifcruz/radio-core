class Chopper:


    def __init__(self, size: int, chunk_size: int):
        self.__size = size
        self.__chunk_size = chunk_size

        if (self.__size % self.__chunk_size) != 0:
            raise ValueError('cannot evenly divide array by chunk size')


    def chop(self, arr):
        for i in range(self.__size//self.__chunk_size):
            yield arr[self.__chunk_size*i:self.__chunk_size*(i+1)]


    def get_to_da_choppa(self):
        print('https://www.youtube.com/watch?v=Xs_OacEq2Sk')

