import copy


class Container:
    def __init__(self, values=None):
        if values == None:
            values = []
        self._values = values

    def append(self, value):
        self._values.append(value)
        return self

    def copy(self):
        return self.__class__(copy.copy(self._values))

    def __add__(self, value):
        return self.copy().append(value)

    def __repr__(self):
        return f"<Container({self._values})>"


values = "hello Python".split()
c = Container(values)
