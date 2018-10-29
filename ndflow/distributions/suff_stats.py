class SuffStats(object):
    def _suff_stats(self, X):
        raise NotImplementedError

    def add(self, X):
        values = self._suff_stats(X)
        for key, value in values.items():
            self.__dict__[key] += value

    def remove(self, X):
        values = self._suff_stats(X)
        for key, value in values.items():
            self.__dict__[key] -= value

    def __iadd__(self, X):
        self.add(X)
        return self

    def __isub__(self, X):
        self.remove(X)
        return self
