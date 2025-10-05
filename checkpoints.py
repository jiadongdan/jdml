from __future__ import annotations
import os, re, pathlib, torch
from typing import Optional, Tuple

def save_checkpoint(model, optimizer, epoch, train_loss, train_acc, test_loss, test_acc, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc
    }, filename)
    print(f"Checkpoint saved: {filename}")

def has_checkpoints(folder_name: str = "checkpoints") -> bool:
    os.makedirs(folder_name, exist_ok=True)
    return any(f.endswith((".pth", ".pt")) for f in os.listdir(folder_name))

def load_latest_checkpoint(model, optimizer, folder_name="checkpoints", map_location="cpu",
                           prefer="epoch", strict=True) -> Optional[Tuple[object, object, int]]:
    ckpt_dir = pathlib.Path(folder_name)
    if not ckpt_dir.exists():
        print(f"No folder named {folder_name} found.")
        return None
    if prefer in ("last","best"):
        p = ckpt_dir / f"{prefer}.pth"
        if p.is_file():
            ckpt = torch.load(p, map_location=map_location)
            model.load_state_dict(ckpt["model_state_dict"], strict=strict)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            e = int(ckpt.get("epoch", 0))
            print(f"Loaded checkpoint ({prefer}): {p.name}")
            return model, optimizer, e + 1
    pat = re.compile(r"^epoch_(\d+)\.(?:pth|pt)$")
    candidates = [(int(m.group(1)), p) for p in ckpt_dir.iterdir()
                  if p.is_file() and (m:=pat.match(p.name))]
    if not candidates:
        print("No checkpoint files found.")
        return None
    candidates.sort(key=lambda t: t[0])
    e_label, latest = candidates[-1]
    ckpt = torch.load(latest, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    e = int(ckpt.get("epoch", e_label))
    print(f"Loaded checkpoint: {latest.name}")
    return model, optimizer, e + 1
