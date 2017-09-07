import torch
from torch.autograd import Variable

dtype = torch.FloatTensor

N, D_in, H, D_out = 64, 1000, 100, 10

# create random tensors to hold input and outputs and wrap them in variables
x = Variable(torch.randn(N, D_in).type(dtype), requires_grad = False)
y = Variable(torch.randn(N, D_out).type(dtype), requires_grad = False)
# setting require_grad = False indicates that we don't want the
# gradients to be calculated

# create random Tensors for weights and wrap them in variables
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad = True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad = True)

learning_rate = 1e-6
for t in range(500):
    # perform the forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # compute and print the loss using the operations on variables
    loss = (y_pred - y).pow(2).sum()
    print(t, loss)
    
    # manually zero the gradients before running the backward pass
    w1.grad.data.zero()
    w2.grad.data.zero()

    # we use the autograd to compute the backward pass. 
    loss.backward()

    # update the weight values
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data














