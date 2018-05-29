from hyperopt import  hp,fmin, rand, tpe, space_eval

def q (args) :
    x, y = args
    return x**2-2*x+1 + y**2

space = [hp.randint('x', 5), hp.randint('y', 5)]

best = fmin(q,space,algo=rand.suggest,max_evals=10)

print(best)