import heapq
from numpy.linalg import norm

def astar(graph, start, goal, gdistance = None, hdistance = None):
     '''
     Compute A* path for the input graph connecting start to goal.
     Edges might be oriented.
     gdistance is the distance function between two nodes of the graph.
     hdistance is the heuristic distance function typically called between 
     any arbitrary nodes of the graph and the goal.
     '''
     frontier         = [ (0, start) ]    # frontier should be a sorted heap 
                                          # (use push and pop methods.
     cameFrom         = { start: None }   # represent a tree.
     costToHere       = { start: 0    }   # cost from start to current node.

     # Graph distance
     if gdistance is None:          gdistance = lambda i1,i2: graph.edgeCost[i1,i2]
     # Heuristic distance
     if hdistance is None:          hdistance = lambda i1,i2: norm(graph.x[i1]-graph.x[i2])

     # Compute the path from leave to root in a tree
     pathFromTree = lambda tree,path,start,goal: \
         [start,]+path if goal==start \
         else pathFromTree(tree,[goal,]+path,start,tree[goal])

     # Push and pop in a sorted heap
     pop  = lambda heap: heapq.heappop(heap)[1]
     push = lambda heap,item,cost: heapq.heappush(heap,(cost,item))

     # A* make groth a set initially containing only the start while
     # maintaining a list of the nodes a the frontier of this set.
     # The set is iterativelly extended by heuristcally choosing in 
     # the frontier list, until the frontier reaches the goal.
     while len(frontier)>0:
          cur = pop(frontier)                   # Pick the (heuristic) max of the frontier
          if cur == goal:                       # If it is the goal: stop
              return pathFromTree(cameFrom,[], # Return the trajectory from tree <camFrom>
                                    start,goal)  # root to goal.

          for nex in graph.children[cur]:       # Expand the frontier to cur node childrens
               curCost = costToHere[cur] + gdistance(cur, nex)          # Exact cost to nex
               if nex not in costToHere or curCost < costToHere[nex]:
                    # If nex is not yet explored or if a shortest path to nex has been found
                    costToHere[nex] = curCost                           # Set cost-to-here.
                    push(frontier,nex, curCost + hdistance(goal, nex))  # Add nex to the sorted frontier
                    cameFrom[nex] = cur                                 # Add nex to tree.
                    
     # If arriving here: start and goal are not in the same connex component
     # of the graph. Return empty path.
     return []
