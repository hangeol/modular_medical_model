import torch
def train(model, batches, optimizer):
    device = next(model.module.parameters()).device
    model.train()
    model.module.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    for batch in batches:
        batch = {name: value.to(device) for name, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(**batch) / batch["labels"][:, 1:].ne(-100).sum()
        loss.backward()
        optimizer.step()
