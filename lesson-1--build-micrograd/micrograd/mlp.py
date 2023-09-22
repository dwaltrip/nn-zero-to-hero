import random
from value import Value


class Neuron:
    def __init__(self, nin):
        self.w = [rand_val() for _ in range(nin)]
        self.b = rand_val()

    def __call__(self, x):
        # w * x + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out
    
    def parameters(self):
        return self.w + [self.b]


def rand_val():
    return Value(random.uniform(-1, 1))


class Layer:
    def __init__(self, nin, nout):
        self._size = (nin, nout)
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [param
            for neuron in self.neurons
            for param in neuron.parameters()
        ]


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)
        ]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [param
            for layer in self.layers
            for param in layer.parameters()
        ]


def mse_loss(ys, ypred):
    return sum(
        ((yout - ygt)**2 for ygt, yout in zip(ys, ypred))
    )
    

# -----------------------------------------------------------------------------

def main():
    x = [2.0, 3.0, -1.0]
    n = MLP(3, [4, 4, 1])
    output = n(x)
    print(output)
    output.backward()
    output.build_viz().view()


def main2():
    LEARN_RATE = 0.1
    # LEARN_RATE = 0.01

    net = MLP(3, [4, 4, 1])
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    # simple training loop
    for i in range(40):
        # forward pass
        ypred = [net(x) for x in xs]
        loss = mse_loss(ys=ys, ypred=ypred)

        # reset grads, backward pass
        for p in net.parameters():
            p.grad = 0.0
        loss.backward()

        # update parameters
        for p in net.parameters():
            p.data += -1 * LEARN_RATE * p.grad

        print(f'epoch {i+1} - loss: {loss.data}')
    
    print()
    print('final ypred:', [round(y.data, 4) for y in ypred])


if __name__ == '__main__':
    # main()
    main2()
