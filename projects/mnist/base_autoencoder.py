from torch import nn

from projects.mnist.representation_splitting import get_mlp


class Autoencoder(nn.Module):
    def __init__(self, hidden_layer_sizes, hidden_size, **kwargs):
        # kwargs added so irrelevant arguments can be passed without error
        super(Autoencoder, self).__init__()
        assert type(hidden_layer_sizes) is list, "Must pass list of layer sizes"

        self.hidden_size = hidden_size
        input_size = 28 * 28

        # Create encoders and decoders
        layer_sizes = [input_size] + hidden_layer_sizes + [self.hidden_size]
        self.encoder = get_mlp(layer_sizes, final_layer_has_bias=False)
        self.decoder = get_mlp(layer_sizes[::-1], final_layer_has_bias=True)

    def forward(self, x):
        batch_size = x.shape[0]
        encoding = self.encoder(x.reshape((batch_size, 784)))
        out = self.decoder(encoding).reshape((batch_size, 1, 28, 28))
        return out, encoding


def calculate_loss(batch, model):
    device = next(model.parameters()).device
    img, labels = [d.to(device) for d in batch]
    out, encodings = model(img)  # encodings.shape = (batch, hidden_size)

    losses = {
        "Decoder": ((out - img).abs()).mean(),
        "Encoder L1 penalty": encodings.abs().mean(dim=0).sum(),
    }
    return losses
