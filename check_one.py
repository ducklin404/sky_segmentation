import torch
from PIL import Image
import torchvision.transforms.functional as TF
from mobiletunet import MobileUNetLite

device = "cuda" if torch.cuda.is_available() else "cpu"
TARGET = 224

# load model (adjust path as needed)
model = MobileUNetLite().to(device)
ckpt = torch.load("./checkpoints/best_finetuned.pth", map_location=device)
model.load_state_dict(ckpt["state_dict"])
model.eval()

def resize_fit_and_pad(img: Image.Image, target: int = TARGET):
    # Resize so the image fits inside target x target (both dims <= target),
    # then pad to exactly target x target (letterbox)
    w, h = img.size
    scale = target / max(w, h)  # <-- fit inside
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img_resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_left = (target - new_w) // 2
    pad_top  = (target - new_h) // 2
    pad_right = target - new_w - pad_left
    pad_bottom = target - new_h - pad_top

    # TF.pad expects (left, top, right, bottom)
    img_padded = TF.pad(img_resized, (pad_left, pad_top, pad_right, pad_bottom))

    padding = (pad_left, pad_top, pad_right, pad_bottom)
    return img_padded, padding, (new_w, new_h)

def infer_full_image(image_path, threshold=0.5):
    orig = Image.open(image_path).convert("RGB")
    W, H = orig.size

    img_pad, padding, (res_w, res_h) = resize_fit_and_pad(orig, TARGET)
    tensor = TF.to_tensor(img_pad).unsqueeze(0).to(device)  # [1,3,TARGET,TARGET]

    with torch.no_grad():
        logits = model(tensor)            # expect [1,1,TARGET,TARGET] (or similar)
        probs = torch.sigmoid(logits)    # [1,1,TARGET,TARGET]
        probs = probs.squeeze(0).squeeze(0)  # [TARGET, TARGET]

    # Remove the padding we added
    pl, pt, pr, pb = padding
    # compute crop box on the probs tensor
    top = pt
    left = pl
    bottom = TARGET - pb
    right = TARGET - pr
    mask_cropped = probs[top:bottom, left:right]  # shape -> [res_h, res_w]

    # Resize mask back to original image size (H, W)
    mask_resized = TF.resize(mask_cropped.unsqueeze(0), (H, W), interpolation=TF.InterpolationMode.BILINEAR)
    mask_resized = mask_resized.squeeze(0)
    mask_binary = (mask_resized > threshold).float()

    return mask_binary.cpu()  # tensor shape [H, W], values 0/1

# Example usage
if __name__ == "__main__":
    mask = infer_full_image(r"D:\Downloads\bdd100k\images\00a360bd-27ccb1dd.jpg")
    import torchvision.utils as vutils
    vutils.save_image(mask.unsqueeze(0), "mask_fullsize_fixed.png")
    print("Saved mask_fullsize_fixed.png")
