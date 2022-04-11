class Result():
    def __init__(self, objVal, statistics, tour):
        self._objVal = objVal
        self._statistics = statistics
        self._tour = tour
    
    @property
    def objVal(self):
        return self._objVal
    
    @property
    def statistics(self):
        return self._statistics
    
    @property
    def tour(self):
        return self._tour
