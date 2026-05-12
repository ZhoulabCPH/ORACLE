import torch
from tqdm import tqdm


@torch.no_grad()
def eval_epoch_no_tta(model, loader, criterion, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for x, y in tqdm(loader, desc="Validation", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)

        total_loss += loss.item() * x.size(0)
        all_probs.append(probs.detach().cpu())
        all_labels.append(y.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, torch.cat(all_probs), torch.cat(all_labels)
