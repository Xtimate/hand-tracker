import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dataset import FreiHAND
from model import HandNet, decode, simcc_loss

@torch.no_grad()
def evaluate(model, loader, device, k=2):
    model.eval()
    hits, total, err_sum = 0, 0, 0.0
    for img, kp in loader:
        img, kp = img.to(device), kp.to(device)
        px, py = model(img)
        dist = (decode(px, py, k) - kp).norm(dim=-1)
        span = (kp.amax(1) - kp.amin(1)).amax(-1, keepdim=True).clamp_min(1.0)
        hits += (dist / span < 0.2).sum().item()
        total += dist.numel()
        err_sum += dist.sum().item()
    return hits / total, err_sum / total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"device={device} amp={use_amp}")

    if args.overfit:
        base = FreiHAND(args.root, train=False, size=args.size)
        train_ds = val_ds = Subset(base, range(args.overfit))
    else:
        train_ds = FreiHAND(args.root, train=True, size=args.size)
        val_ds = FreiHAND(args.root, train=False, size=args.size)

    kw = dict(num_workers=args.workers, pin_memory=use_amp, persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.bs, **kw)

    model = HandNet(size=args.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    best = 0.0

    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        for img, kp in train_dl:
            img, kp = img.to(device), kp.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                px, py = model(img)
            loss = simcc_loss(px.float(), py.float(), kp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()

        pck, err = evaluate(model, val_dl, device)
        print(f"epoch {epoch + 1:3d}    loss {run / len(train_dl):.4f}  "
            f"PCK@0.2 {pck:.3f}     err {err:.2f}px")
        if pck >= best:
            best = pck
            torch.save({"model": model.state_dict(), "size": args.size},
                out / "best.pt")

if __name__ == "__main__":
    main()
