import argparse
from pathlib import Path

import torch
import time
from torch._decomp.decompositions import binary_cross_entropy_with_logits
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from dataset import FreiHAND
from model import HandNet, decode, simcc_loss

PRES_W = 10.0

def fmt(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

@torch.no_grad()
def evaluate(model, loader, device, k=2):
    model.eval()
    hits, total, err_sum, pres_ok, n = 0, 0, 0.0, 0, 0
    for img, kp, pres in loader:
        img, kp, pres = img.to(device), kp.to(device), pres.to(device)
        px, py, logit = model(img)
        pres_ok += ((logit.squeeze(-1) > 0).float() == pres).sum().item()
        n += pres.numel()
        pos = pres > 0.5
        if pos.any():
            dist = (decode(px, py, k) - kp).norm(dim=-1)[pos]
            span = (kp.amax(1) - kp.amin(1)).amax(-1, keepdim=True).clamp_min(1.0)[pos]
            hits += (dist / span < 0.2).sum().item()
            total += dist.numel()
            err_sum += dist.sum().item()
    return hits / total, err_sum / total, pres_ok / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--bg-dir", default=None)
    ap.add_argument("--limit", type=int, default=0)
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
        val_bg = None
    else:
        train_ds = FreiHAND(args.root, train=True, size=args.size, bg_dir=args.bg_dir)
        val_ds = FreiHAND(args.root, train=False, size=args.size)
        val_bg = FreiHAND(args.root, train=False, size=args.size, bg_dir=args.bg_dir, bg_prob=1.0, neg_prob=0.3) if args.bg_dir else None
        if args.limit:
            train_ds = Subset(train_ds, range(args.limit))
            val_ds = Subset(val_ds, range(args.limit // 4))
            val_bg = Subset(val_bg, range(args.limit // 4)) if val_bg is not None else None

    kw = {
        "num_workers": args.workers,
        "pin_memory":use_amp,
        "persistent_workers": args.workers > 0,
    }

    train_dl = DataLoader(train_ds, args.bs, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.bs, **kw)
    val_bg_dl = DataLoader(val_bg, args.bs, **kw) if val_bg is not None else None

    model = HandNet(size=args.size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) # type: ignore

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    best = 0.0

    total_steps = args.epochs * len(train_dl)
    done_steps = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        run = 0.0
        bar = tqdm(train_dl, desc=f"epoch  {epoch + 1}/{args.epochs}", leave=False, dynamic_ncols=True)
        for img, kp, pres in bar:
            img, kp, pres = img.to(device), kp.to(device), pres.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                px, py, logit = model(img)
            loss = (simcc_loss(px.float(), py.float(), kp, mask=pres) + PRES_W * F.binary_cross_entropy_with_logits(logit.float().squeeze(-1), pres))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            l = loss.item()
            run += l
            done_steps += 1
            eta = (time.time() - t0) / done_steps * (total_steps - done_steps)
            bar.set_postfix(loss=f"{l:.3f}", total_left=fmt(eta))

        pck, err, _ = evaluate(model, val_dl, device)
        msg = f"epoch {epoch + 1:3d}  loss {run / len(train_dl):.4f}  PCK@0.2 {pck:.3f}  err {err:.2f}px"
        score = pck
        if val_bg_dl:
            pck_bg, err_bg, acc_bg = evaluate(model, val_bg_dl, device)
            msg += f"  | bg PCK {pck_bg:.3f}  err {err_bg:.2f}px  pres {acc_bg:.3f}"
            score = pck_bg
        print(msg, flush=True)
        if score >= best:
            best = score
            torch.save({"model": model.state_dict(), "size": args.size}, out / "best.pt")

if __name__ == "__main__":
    main()
