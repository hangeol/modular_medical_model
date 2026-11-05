import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import Gemma4ForCausalLM


def load_model(path, dtype, device):
    return Gemma4ForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        key_mapping={r"^model\.language_model\.": "model."},
    )


def native_logits(model, hidden):
    logits = model.lm_head(hidden)
    cap = model.config.final_logit_softcapping
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


class ModularMedicalModel(nn.Module):
    def __init__(self, backbone, module, alpha=1.0, chunk_size=128):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.module = module
        self.alpha = alpha
        self.chunk_size = chunk_size

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def chunk_loss(self, backbone_hidden, module_hidden, labels):
        with torch.no_grad():
            backbone_logits = native_logits(self.backbone, backbone_hidden)
        module_logits = native_logits(self.module, module_hidden)
        logits = backbone_logits.to(module_logits.dtype) + self.alpha * module_logits
        return F.cross_entropy(logits.float(), labels, ignore_index=-100, reduction="sum")

    def forward(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            backbone_hidden = self.backbone.model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).last_hidden_state
        module_hidden = self.module.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        backbone_hidden = backbone_hidden[:, :-1].reshape(-1, backbone_hidden.shape[-1])
        module_hidden = module_hidden[:, :-1].reshape(-1, module_hidden.shape[-1])
        targets = labels[:, 1:].reshape(-1)
        losses = []
        for start in range(0, targets.numel(), self.chunk_size):
            end = start + self.chunk_size
            losses.append(checkpoint(
                self.chunk_loss,
                backbone_hidden[start:end],
                module_hidden[start:end],
                targets[start:end],
                use_reentrant=False,
            ))
        return torch.stack(losses).sum()
