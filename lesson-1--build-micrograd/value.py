from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = 0.0

        self._prev = set(_children)
        self._op = _op

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
