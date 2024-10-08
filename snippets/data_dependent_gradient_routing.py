import torch


def forward(self, x: torch.Tensor, gradient_mask: list[torch.Tensor]):
    """
    A sample forward pass with gradient routing.

    Most of the work required to implement gradient routing in full
    generality is in defining the gradient mask. The gradient mask is
    a list of Tensors, with gradient_mask[i] corresponding to layer i's
    activations and having shape = (batch_size, layer_i_output_size).
    The layer i activation mask for data point j is given by
    gradient_mask[i][j].

    For a specific application, there may be an easier way to apply
    gradient routing, depending on where/when you want to apply the mask.
    For example, for representation splitting of MNIST digits, the
    gradient masking was applied only on the encoder output, and it was
    always either [1,...,1,0,...,0] or [0,...,0,1,...,1].
    """
    out = x
    for layer, mask in zip(self.layers, gradient_mask):
        activation = layer(out)
        out = mask * activation + (1 - mask) * activation.detach()
    return out
