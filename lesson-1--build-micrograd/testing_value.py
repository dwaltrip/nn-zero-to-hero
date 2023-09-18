from value import Value


def testing_backward():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')

    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    L.grad = 1.0
    L.backward(verbose=True)

    L.build_viz().view()


if __name__ == '__main__':
    testing_backward()
