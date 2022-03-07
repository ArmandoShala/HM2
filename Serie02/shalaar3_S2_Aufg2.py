import sympy as sp

x1, x2, x3 = sp.symbols('x1, x2, x3')


def a1():
    f1 = 5 * x1 * x2
    f2 = x1 ** 2 * x2 ** 2 + x1 + 2 * x2

    f = sp.Matrix([f1, f2])
    x = sp.Matrix([x1, x2])

    Df = f.jacobian(x)
    print("jacobian:")
    print(Df)

    Df0 = Df.subs([(x1, 1), (x2, 2)])
    print(Df0)
    print(Df0.evalf())


def a2():
    f1 = sp.log(x1 ** 2 + x2 ** 2) + x3 ** 2
    f2 = sp.exp(x2 ** 2 + x2 ** 2) + x1 ** 2
    f3 = 1 / (x3 ** 2 + x1 ** 2) + x2 ** 2

    f = sp.Matrix([f1, f2, f3])
    x = sp.Matrix([x1, x2, x3])

    Df = f.jacobian(x)
    print("jacobian:")
    print(Df)
    Df0 = Df.subs([(x1, 1), (x2, 2), (x3, 3)])
    print(Df0)

    print(Df0.evalf())


a1()
print("===========================")
a2()
