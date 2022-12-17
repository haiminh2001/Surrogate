class AbstractTask():
    def __init__(self, *args, **kwargs) -> None:
        pass
    def __eq__(self, __o: object) -> bool:
        if self.__repr__() == __o.__repr__():
            return True
        else:
            return False
    def decode(self, x):
        pass
    def __call__(self, x):
        pass

    @staticmethod
    def func(x):
        pass

    


