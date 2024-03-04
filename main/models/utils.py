import torch
import torch.nn as nn

from . import hybrid
from . import vit
from . import transformer


class Model(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.args = args

    def data_parallel(self, x: torch.Tensor, device_ids, output_device=None, **kwargs):
        if not device_ids or len(device_ids) == 1:
            return self(x, **kwargs)
        if output_device is None:
            output_device = device_ids[0]
        replicas = nn.parallel.replicate(self, device_ids)
        # Slices tensors into approximately equal chunks and distributes them across given GPUs.
        inputs = nn.parallel.scatter(x, device_ids)
        # Duplicates references to objects that are not tensors.
        kwargs = nn.parallel.scatter(kwargs, device_ids)
        replicas = replicas[:len(inputs)]
        kwargs = kwargs[:len(inputs)]
        outputs = nn.parallel.parallel_apply(replicas, inputs, kwargs)
        return nn.parallel.gather(outputs, output_device).mean()

    def forward(self, x: torch.Tensor, tgt_seq: torch.Tensor,  **kwargs):
        encoded = self.encoder(x)
        out = self.decoder(tgt_seq, context=encoded, **kwargs)
        return out

    @torch.no_grad()
    def generate(self, x: torch.Tensor, temperature: float = 0.25):
        return self.decoder.generate((torch.LongTensor([self.args.bos_token]*len(x))[:, None]).to(x.device), self.args.max_seq_len,
                                     eos_token=self.args.eos_token, context=self.encoder(x), temperature=temperature)


def get_model(encoder_structure, device, model_args):
    if encoder_structure.lower() == 'vit':
        encoder = vit.get_encoder(model_args)
    elif encoder_structure.lower() == 'hybrid':
        encoder = hybrid.get_encoder(model_args)
    else:
        raise NotImplementedError(
            'Encoder structure "%s" not supported.' % encoder_structure)
    decoder = transformer.get_decoder(model_args)
    encoder.to(device)
    decoder.to(device)
    model = Model(encoder, decoder, model_args)

    return model
