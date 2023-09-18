from value import Value


def lol():
    a, b, c = [Value(v, label=k) for k,v in dict(a=2.0, b=-3.0, c=10.0).items()]
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    L.grad = 1.0
    f.grad = d.data
    d.grad = f.data

    c.grad = d.grad
    e.grad = d.grad

    # chain rule!
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    L.build_viz().view()


def foo():
    h = 0.001

    a, b, c = [Value(v, label=k) for k,v in dict(a=2.0, b=-3.0, c=10.0).items()]
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L1 = L.data

    a, b, c = [Value(v, label=k) for k,v in dict(a=2.0, b=-3.0, c=10.0).items()]
    b.data += h
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L2 = L.data

    print((L2 - L1) / h)



if __name__ == '__main__':
    # lol()
    foo()
