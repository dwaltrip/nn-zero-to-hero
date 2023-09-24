import sys

from micrograd.value import Value


# H needs to be much smaller than the grad check treshhold,
#   as expoential functions can exaggerate small diffs
DERIVATIVE_H = 10 ** (-5)
print('DERIVATIVE_H:', DERIVATIVE_H)
GRAD_CHECK_TRESHHOLD = 0.01

class TestError(Exception):
    pass

# -----------------------------------------------------------------------------

def test_add():
    def calc1(a=None, b=None):
        return a + b
    check_grads(fn=calc1, a=Value(2.0), b=Value(3.0))

    def calc2(a=None):
        return a + a
    check_grads(fn=calc2, a=Value(2.0))

    def calc3(a=None):
        return 2 + a
    check_grads(fn=calc3, a=Value(2.0))

    def calc4(a=None):
        return (a + 2) + (a + 3)
    check_grads(fn=calc4, a=Value(2.0))


def test_mult():
    def calc1(a=None, b=None):
        return a * b
    check_grads(fn=calc1, a=Value(2.0), b=Value(3.0))

    def calc2(a=None):
        return a * a
    check_grads(fn=calc2, a=Value(2.0))

    def calc3(a=None):
        return 2 * a
    check_grads(fn=calc3, a=Value(2.0))

    def calc4(a=None):
        return (a * 2) * (a * 1.5)
    check_grads(fn=calc4, a=Value(2.0))


def test_sub():
    def calc1(a=None, b=None):
        return a - b
    check_grads(fn=calc1, a=Value(2.0), b=Value(3.0))

    def calc2(a=None):
        return (2 * a) - a
    check_grads(fn=calc2, a=Value(2.0))

    def calc3(a=None):
        return 5 - a
    check_grads(fn=calc3, a=Value(2.0))


def test_div():
    def calc1(a=None, b=None):
        return a / b
    check_grads(fn=calc1, a=Value(2.0), b=Value(3.0))

    def calc2(a=None):
        return (2 * a) / a
    check_grads(fn=calc2, a=Value(2.0))

    def calc3(a=None):
        return 5 / a
    check_grads(fn=calc3, a=Value(2.0))

    def calc4(a=None, b=None):
        return ((a + b) - 1) / ((a + b) + 1)
    check_grads(fn=calc4, a=Value(2.0), b=Value(3.0))

    def calc5(a=None, b=None):
        c = a + b
        return c / (c + 1)
    check_grads(fn=calc5, a=Value(2.0), b=Value(3.0))


def test_pow():
    def calc1(a=None):
        return a ** 2.5
    check_grads(fn=calc1, a=Value(2.0))

    def calc2(a=None):
        return (a ** 2) + (a ** 3)
    check_grads(fn=calc2, a=Value(2.0))

    def calc3(a=None, b=None):
        return (a ** 2) * b
    check_grads(fn=calc3, a=Value(2.0), b=Value(3.0))

    def calc4(a=None, b=None):
        return (a + b) ** 2
    check_grads(fn=calc3, a=Value(2.0), b=Value(3.0))


def test_exp():
    def calc1(a=None):
        return a.exp()
    check_grads(fn=calc1, a=Value(2.0))

    def calc2(a=None):
        return (2 * a).exp()
    check_grads(fn=calc2, a=Value(2.0))

    def calc3(a=None):
        return a.exp() + (2 * a).exp()
    check_grads(fn=calc3, a=Value(2.0))

    def calc4(a=None, b=None):
        return a.exp() * b
    check_grads(fn=calc4, a=Value(2.0), b=Value(3.0))

    def calc5(a=None, b=None):
        return (a + b).exp()
    check_grads(fn=calc5, a=Value(2.0), b=Value(3.0))

    def calc6(a=None):
        return (a.exp() + 1) / (a.exp() - 1)
    check_grads(fn=calc6, a=Value(2.0))


def test_all_the_things():
    def calc1(x1=None, x2=None, w1=None, w2=None, b=None):
        n = (x1 * w1) + (x2 * w2) + b
        return (n - 1) / (n + 1)
    check_grads(fn=calc1,
        x1 = Value(2.0),
        x2 = Value(0.0),
        w1 = Value(-3.0),
        w2 = Value(1.0),
        b = Value(6.881373),
    )

    def calc2(x1=None, x2=None, w1=None, w2=None, b=None):
        n = (x1 * w1) + (x2 * w2) + b
        n_e = (2 * n).exp()
        return (n_e - 1) / (n_e + 1)
    check_grads(fn=calc2,
        x1 = Value(2.0),
        x2 = Value(0.0),
        w1 = Value(-3.0),
        w2 = Value(1.0),
        b = Value(6.881373),
    )

    # This expression and calc1's were what gave me incorrect grads originally
    def calc3(x1=None, x2=None, w1=None, w2=None, b=None):
        n = (x1 * w1) + (x2 * w2) + b
        return ((2 * n).exp() - 1) / ((2 * n).exp() + 1)
    check_grads(fn=calc3,
        x1 = Value(2.0),
        x2 = Value(0.0),
        w1 = Value(-3.0),
        w2 = Value(1.0),
        b = Value(6.881373),
    )

# -----------------------------------------------------------------------------

def check_grads(fn, debug=False, **vals):
    def copy_of_vals():
        return { key: val.copy() for key, val in vals.items() }

    failed_val_names = []
    for name in vals.keys():
        vals_1 = copy_of_vals()
        y1 = fn(**vals_1)

        vals_2 = copy_of_vals()
        vals_2[name].data += DERIVATIVE_H
        y2 = fn(**vals_2)

        y1.backward()
        x1_grad = vals_1[name].grad
        x2_grad = (y2.data - y1.data) / DERIVATIVE_H
        if abs(x1_grad - x2_grad) > GRAD_CHECK_TRESHHOLD:
            failed_val_names.append(name)
        
        if debug:
            print(f'\t{fn.__name__}, var = {name} --', ', '.join([
                f'{name}1 = {vals_1[name].data}',
                f'{name}2 = {vals_2[name].data}',
                f'y1 = {y1.data:0.4f}',
                f'y2 = {y2.data:0.4f}',
                f'{name}1_grad = {x1_grad:0.4f}',
                f'{name}2_grad = {x2_grad:0.4f}',
                f'diff = {abs(x1_grad - x2_grad):0.5f}',
            ]))

    if len(failed_val_names) > 0:
        err_msg = ' '.join([
            f'grad check failed -- fn: {fn.__name__}',
            '-- failed vars:',
            ', '.join(failed_val_names),
        ])
        raise TestError(err_msg)

# -----------------------------------------------------------------------------

TESTS = [
    test_add,
    test_mult,
    test_sub,
    test_div,
    test_pow,
    test_exp,
    test_all_the_things,
]

if __name__ == '__main__':
    failures = []

    for test in TESTS:
        # print(f'running {test.__name__}...', end=' ')
        print(f'running {test.__name__}...')
        try:
            test()
            print('\t' + 'success')
        except TestError as err:
            failures.append(err)
            print('\t' + 'FAILED')
    
    print()
    print('Failures:')
    for err in failures:
        print('\t' + str(err))
    if len(failures) == 0:
        print('\t' + 'None! :)')
