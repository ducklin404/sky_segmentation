import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from dataset import SkyPatchDataset
from mobiletunet import MobileUNetLite

# utils / metrics
def compute_iou(pred_logits, mask, thresh=0.5, eps=1e-7):
    # pred_logits: [B,1,H,W] or [B,H,W]
    if pred_logits.dim() == 4 and pred_logits.shape[1] == 1:
        pred = torch.sigmoid(pred_logits).squeeze(1)
    else:
        pred = torch.sigmoid(pred_logits)

    pred_bin = (pred >= thresh).to(torch.uint8)
    mask_bin = mask.to(torch.uint8)

    inter = (pred_bin & mask_bin).sum(dim=(1,2)).float()
    union = (pred_bin | mask_bin).sum(dim=(1,2)).float()
    iou = (inter + eps) / (union + eps)
    return iou.mean().item()

def save_ckpt(state, path):
    torch.save(state, path)

# training functions
def train_one_epoch(model: MobileUNetLite, loader, optimizer, device, loss_fn, scaler=None):

    model.train()
    total_loss = 0.0
    total_iou = 0.0
    n = 0

    for imgs, masks in tqdm(loader, desc="train", leave=False):
        # move batch to fit device
        imgs = imgs.to(device)                
        masks = masks.to(device).float()         

        # clear old gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(imgs) # [b,1,112,112]
        # bcewithlogitsloss expects [b,h,w]
        logits = outputs.squeeze(1) # [b,112,112]

        # mixed precision branch 
        if scaler is not None:
            with torch.amp.autocast():
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # standard fp32 training
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()

        # accumulate weighted loss and iou
        total_loss += loss.item() * imgs.size(0)
        total_iou += compute_iou(outputs.detach(), masks)
        n += imgs.size(0)

    # return epoch-averaged loss and mean iou
    return total_loss / n, total_iou / (len(loader))


def validate(model: MobileUNetLite, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    n = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="val", leave=False):
            imgs = imgs.to(device)
            masks = masks.to(device).float()
            outputs = model(imgs)
            logits = outputs.squeeze(1)
            loss = loss_fn(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            total_iou += compute_iou(outputs, masks)
            n += imgs.size(0)
    return total_loss / n, total_iou / (len(loader))

# main training routine
def run_training(
    train_dataset,
    val_dataset,
    model: MobileUNetLite,
    device='cuda',
    batch_size=16,
    freeze_epochs=10,
    unfreeze_epochs=20,
    lr=1e-3,
    lr2=1e-4,
    weight_decay=1e-5,
    pos_weight=None,
    use_amp=True,
    checkpoint_dir='./checkpoints'
):
    # both 2 training step here
    # first phase to train the decoder
    # second phase to fine tune both (unfreeze the encoder)    
    
    
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # loss: BCEWithLogits expects logits; pos_weight vector adjusts for class imbalance
    if pos_weight is not None:
        pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    # Phase 1: freeze encoder
    # freeze encoder weights
    for p in model.encoder.parameters():
        p.requires_grad = False

    # optimizer should only include parameters that require grad (decoder + final)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(opt_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    scaler = torch.amp.GradScaler() if (use_amp and device.type == 'cuda') else None

    best_val_iou = -1.0
    print(f"Phase 1: training decoder (encoder frozen) for {freeze_epochs} epochs")
    for epoch in range(1, freeze_epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler)
        val_loss, val_iou = validate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        print(f"[F1] Epoch {epoch}/{freeze_epochs}  train_loss={train_loss:.4f} train_iou={train_iou:.4f}  val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        # save best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_ckpt({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_iou': val_iou
            }, os.path.join(checkpoint_dir, 'best_frozen.pth'))

    # Phase 2: unfreeze and fine-tune
    print("Phase 2: unfreezing encoder and fine-tuning full model")
    for p in model.encoder.parameters():
        p.requires_grad = True

    # new optimizer for all params 
    optimizer = Adam(model.parameters(), lr= lr2, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    for epoch in range(1, unfreeze_epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler)
        val_loss, val_iou = validate(model, val_loader, device, loss_fn)
        scheduler.step(val_loss)

        print(f"[F2] Epoch {epoch}/{unfreeze_epochs}  train_loss={train_loss:.4f} train_iou={train_iou:.4f}  val_loss={val_loss:.4f} val_iou={val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_ckpt({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_iou': val_iou
            }, os.path.join(checkpoint_dir, 'best_finetuned.pth'))

    print("Training done. Best val IoU:", best_val_iou)
    return model


if __name__ == "__main__":
    # paths
    train_img_dir = "dataset/train/images"
    train_mask_dir = "dataset/train/labels"
    val_img_dir = "dataset/val/labels"
    val_mask_dir = "dataset/val/labels"

    # instantiate your dataset (assumes it returns images [3,224,224] and masks [112,112])
    train_ds = SkyPatchDataset(train_img_dir, train_mask_dir, img_size=224, stride=112)
    val_ds   = SkyPatchDataset(val_img_dir, val_mask_dir, img_size=224, stride=112)

    # optionally compute pos_weight from train dataset (rough heuristic)
    # compute frequency over a subset to avoid long scan
    def estimate_pos_weight(dataset, max_samples=2000):
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
        total_pos = 0
        total_neg = 0
        seen = 0
        for imgs, masks in loader:
            masks = masks.view(-1)
            total_pos += (masks == 1).sum().item()
            total_neg += (masks == 0).sum().item()
            seen += 1
            if seen > max_samples // 16:
                break
        if total_pos == 0:
            return 1.0
        return max(1.0, total_neg / (total_pos + 1e-6))

    est_pw = estimate_pos_weight(train_ds)
    print("Estimated pos_weight (neg/pos):", est_pw)

    # model
    model = MobileUNetLite(num_classes=1)

    trained_model = run_training(
        train_dataset=train_ds,
        val_dataset=val_ds,
        model=model,
        device='cuda',
        batch_size=16,
        freeze_epochs=8,
        unfreeze_epochs=20,
        lr=1e-3,
        lr2=1e-4,
        weight_decay=1e-5,
        pos_weight=est_pw,
        use_amp=True,
        checkpoint_dir='./checkpoints'
    )
