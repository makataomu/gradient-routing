# %%
import torch

"""
Calling .detach() on a Tensor creates a *new* Tensor object which
wraps the *same* data of the original tensor, and which has
requires_grad = False, disabling gradient tracking.

https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html

Check your understanding on the examples below!
"""

# If a Tensor has requires_grad==True, we track operations on it and
# add them to a computational graph, so that we can run backprop
# later.
x = torch.Tensor([0, 1, 47])
x.requires_grad = True

# So, the result of an operation on a tensor that requires_grad is
# another tensor that requires_grad.
y = x + 3
print(f"{y.requires_grad=}")  # True

# ... unless we've disabled gradient tracking.
with torch.no_grad():
    y2 = x + 3
print(f"{y2.requires_grad=}")  # False

with torch.inference_mode():
    y3 = x + 3
print(f"{y3.requires_grad=}")  # False

# Detaching a Tensor creates a *new* Tensor with gradient tracking
# disabled...
x_detached = x.detach()
print(f"{x_detached.requires_grad=}")  # False

# ... this new object wraps the *same* data-- no memory overhead!
x_detached.data[:] = 3
print(f"{x=}")  # tensor([3., 3., 3.], requires_grad=True)

# Of course, if you use this Tensor in an operation with a Tensor
# that requires_grad, the result will require_grad!
w = x_detached * y
print(f"{w.requires_grad=}")  # True

# Finally, here's a thing that you should probably never do:
# in-place operations that mix-and-match Tensors with different
# gradient-tracking settings.
z = torch.Tensor([0, 1, 2, 3])
z.requires_grad = True
z2 = z * 2
print(f"{z2.requires_grad=}")  # True

subset = z2[:].detach()
z2[:] = subset
print(f"{z2.requires_grad=}")  # True... even though we assigned detached values to it!

# In general, avoiding in-place operations makes it easier to reason about your code.
