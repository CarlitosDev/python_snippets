'''

  From https://github.com/facebookresearch/nevergrad

'''


import nevergrad as ng

def square(x):
    return sum((x - .5)**2)

optimizer = ng.optimizers.OnePlusOne(instrumentation=2, budget=100)
recommendation = optimizer.minimize(square)
print(recommendation)  # optimal args and kwargs