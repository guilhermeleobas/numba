from numba.pycc import CC

cc = CC('my_module')
# Uncomment the following line to print out the compilation steps
cc.verbose = False

@cc.export('multf', 'f8(f8, f8)')
@cc.export('multi1', 'i1(i1, i1)')
@cc.export('multi2', 'i2(i2, i2)')
@cc.export('multi4', 'i4(i4, i4)')
@cc.export('multi8', 'i8(i8, i8)')
def mult(a, b):
    return a * b

@cc.export('square', 'f8(f8)')
def square(a):
    return a ** 2


if __name__ == "__main__":
    # import logging
    # logging.basicConfig(level=logging.INFO)
    import time
    start = time.time()
    cc.compile()
    print(time.time() - start)
    print(mult(2, 3))