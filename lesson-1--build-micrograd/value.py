import math
from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

        self._prev = set(_children)
        self._op = _op

    def copy(self):
        return self.__class__(
            data=self.data,
            label=self.label
        )

    @classmethod
    def convert_if_needed(cls, num_or_val):
        if isinstance(num_or_val, cls):
            return num_or_val
        return cls(num_or_val)

    def __repr__(self):
        return f"Value({self.data})"
    
    def build_viz(self):
        return _draw_dot(self)

    def __add__(self, other):
        other = Value.convert_if_needed(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other

    def __mul__(self, other):
        other = Value.convert_if_needed(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __neg__(self):
        return self * -1
 
    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'Only supporting int, float for now'
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out
    
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += (out.data * out.grad)
        
        out._backward = _backward
        return out

    def backward(self):
        def build_topo_sorted(node):
            topo_sorted = []
            visited = set()

            def recurse(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        recurse(child)
                    topo_sorted.append(v)

            recurse(node)
            return topo_sorted

        topo_sorted = build_topo_sorted(self)
        
        self.grad = 1.0
        for node in reversed(topo_sorted):
            node._backward()


def _draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    nodes, edges = _trace_graph(root)
    for n in nodes:
        uid = str(id(n))
        label_parts = [
            n.label,
            f'data {n.data:.4f}',
            f'grad {n.grad:.4f}',
        ]
        label = '{' + ' | '.join([p for p in label_parts if p]) + '}'
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
