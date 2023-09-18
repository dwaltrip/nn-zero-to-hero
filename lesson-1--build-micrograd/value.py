from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = None

        self._prev = set(_children)
        self._op = _op

    def backward(self, verbose=False):
        if verbose:
            print('Running backward for:', self.label)
        if self.grad is None:
            raise ValueError('Missing grad value.')

        if len(self._prev) == 0:
            # Leaf node -> done with this part of the graph
            return None
        elif len(self._prev) != 2:
            raise ValueError('Should have 0 or 2 inputs.')

        left, right = self._prev
        # Chain rule! Woot woot.
        for (curr, other) in [(left, right), (right, left)]:
            if self._op == '+':
                curr.grad = self.grad
            elif self._op == '*':
                curr.grad = self.grad * other.data
            else:
                raise ValueError(f'Invalid op: {self._op}')
            # Propagate that shiz
            curr.backward(verbose=verbose) 

    def __repr__(self):
        return f"Value({self.data})"
    
    def build_viz(self):
        return _draw_dot(self)
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out


def _draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = _trace_graph(root)
    for n in nodes:
        uid = str(id(n))
        label = '{' + f'{n.label} | data {n.data:.4f} | grad {n.grad:.4f}' + '}'
        dot.node(uid, label=label, shape='record')

        if n._op:
            dot.node(name = uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

def _trace_graph(root):
  """ builds a set of all nodes and edges in a graph """
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges
