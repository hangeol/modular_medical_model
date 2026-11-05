# Modular Medical Model

Minimal Gemma 4 implementation with a frozen backbone and a trainable module.

```bash
pip install -r requirements.txt
```

```python
import torch
from model import ModularMedicalModel, load_model
from train import train

model = ModularMedicalModel(
    load_model(backbone_path, torch.bfloat16, "cuda"),
    load_model(module_path, torch.float32, "cuda"),
)
optimizer = torch.optim.AdamW(model.module.parameters(), lr=learning_rate)
train(model, batches, optimizer)
model.module.save_pretrained("output")
```

Supply model paths, a learning rate, and batches containing `input_ids`, `attention_mask` and `labels` tensors. Both models use the same token IDs; `-100` labels are ignored.
