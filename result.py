class Result():
    def __init__(self, objVal, statistics):
        self._objVal = objVal
        self._statistics = statistics
    
    @property
    def objVal(self):
        return self._objVal
    
    @property
    def statistics(self):
        return self._statistics