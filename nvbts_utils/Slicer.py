import dask.array as da
import numpy as np

class EventTimeSlicer():
    def __init__(self, arr) -> None:    
        """Slices a dask array by time. assumes the time is in the third column while the first column is the x and the second column is the y and the fourth column is the polarity."""
        self.arr = arr

    def slice(self, start, end):
        idx = da.searchsorted(self.arr[:, 2], [start, end])
        return self.arr[idx[0]:idx[1]]
    
