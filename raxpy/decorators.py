

def spec(constraints=None):
    print("SPEC A")
    def cl_f(f):
        print("SPEC B")
        f._constraints = constraints
        return f
    return cl_f

