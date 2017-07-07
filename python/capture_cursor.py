import numpy as np
import matplotlib.pylab as plt

class Cursor(object):
    def __init__(self, ax = None):
        self.ax = ax
        self.x = None
        self.y = None

    def mouse_move(self, event):
        if not event.inaxes:            return
        self.x, self.y = event.xdata, event.ydata

    def connect(self,ax=None):
        if ax is None: ax = self.ax
        if ax is None: ax = plt.gca()
        self.ax = ax
        plt.connect('motion_notify_event', self.mouse_move)

    def __call__(self): return np.matrix([self.x,self.y]).T

cursorCoordinate = Cursor()

__all__ = [ 'cursorCoordinate', ]
