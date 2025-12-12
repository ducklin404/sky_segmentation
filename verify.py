import torch
from mobiletunet import MobileUNetLite
from dataset import SkyPatchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import compute_iou

def evaluate_test(model, loader, device):
    model.eval()
    total_iou = 0.0
    total_loss = 0.0
    n = 0
    loss_fn = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="test"):
            imgs = imgs.to(device)
            masks = masks.to(device).float()

            outputs = model(imgs)
            logits = outputs.squeeze(1)

            loss = loss_fn(logits, masks)
            total_loss += loss.item() * imgs.size(0)

            batch_iou = compute_iou(outputs, masks)
            total_iou += batch_iou
            n += imgs.size(0)

    return total_loss / n, total_iou / len(loader)


if __name__ == "__main__":
    device_str = "cuda"
    ckpt_path = "./checkpoints/best_finetuned.pth"

    model = MobileUNetLite().to(device_str)
    ckpt = torch.load(ckpt_path, map_location=device_str)

    model.load_state_dict(ckpt["state_dict"])
    start_epoch = ckpt["epoch"]
    best_val_iou = ckpt["val_iou"]

    print("Loaded checkpoint from epoch:", start_epoch)

    test_img_dir = "dataset/test/cache_resized/images"
    test_mask_dir = "dataset/test/cache_resized/labels"

    test_ds = SkyPatchDataset(
        test_img_dir,
        test_mask_dir,
        resize_short=1024,
        img_size=224
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=16,
        shuffle=False,
        num_workers=4,  
        pin_memory=True
    )

    test_loss, test_iou = evaluate_test(model, test_loader, device_str)
    print("Test Loss:", test_loss)
    print("Test IoU:", test_iou)
