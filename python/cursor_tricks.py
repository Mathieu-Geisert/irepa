from capture_cursor import *

class FromCursor():
     def __init__(self,prm,grid,env = None):
          self.prm  = prm
          self.grid = grid
          self.hdistance = prm.nearestNeighbor.hdistance
          self.cursorToState = lambda c: np.vstack([c,zero(2)])
          self.display       = None if env is None else env.display
     def node(self):
          '''Get the closest node of the PRM to the cursor coordinate.'''
          x = self.cursorToState(cursorCoordinate())
          return self.prm.nearestNeighbor(x,graph.x)[0]
     def data(self):
          '''Get the closest point of the grid to the cursor coordinate.'''
          x = self.cursorToState(cursorCoordinate())
          return self.prm.nearestNeighbor(x,[d.x0 for d in data])[0]
     def playdata(self):
          idx = self.data()
          print self.grid.data[idx].x0
          for x in self.grid.data[idx].X: 
               self.display(x)
               time.sleep(.1)
