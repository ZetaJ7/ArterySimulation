import taichi as ti

# ti.init(arch=ti.gpu)  # Try to run on GPU

def r_test():
    a = random()
    b = random()
    return a, b

a, b = r_test()
print(a, b)
print("finish")
