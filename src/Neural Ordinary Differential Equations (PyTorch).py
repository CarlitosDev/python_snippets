'''

  Neural Ordinary Differential Equations

  (https://arxiv.org/abs/1806.07366)

  https://github.com/rtqichen/torchdiffeq

  pip3 install git+https://github.com/rtqichen/torchdiffeq

'''


from torchdiffeq import odeint
import torch
func = lambda t,y: t*3

y0 = torch.tensor([6.0])
t = torch.linspace(0., 25., 100)
true_y = odeint(func, y0, t, method='dopri5')