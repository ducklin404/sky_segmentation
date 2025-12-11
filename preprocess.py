import os
import cv2

def preprocess_split(split_dir, resize_short=512):
    img_in = os.path.join(split_dir, "images")
    mask_in = os.path.join(split_dir, "labels")

    img_out = os.path.join(split_dir, "cache_resized", "images")
    mask_out = os.path.join(split_dir, "cache_resized", "labels")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    for name in sorted(os.listdir(img_in)):
        in_img_path = os.path.join(img_in, name)
        in_mask_path = os.path.join(mask_in, os.path.splitext(name)[0] + ".png")

        img = cv2.imread(in_img_path)
        mask = cv2.imread(in_mask_path, 0)

        if img is None or mask is None:
            continue

        h, w = img.shape[:2]
        scale = resize_short / min(h, w)

        new_w, new_h = int(w * scale), int(h * scale)

        img_r = cv2.resize(img, (new_w, new_h))
        mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(img_out, name), img_r)
        cv2.imwrite(os.path.join(mask_out, os.path.splitext(name)[0] + ".png"), mask_r)

    print(f"Done preprocessing: {split_dir}")


# run for train, val, test
preprocess_split("dataset/train")
preprocess_split("dataset/val")
preprocess_split("dataset/test")
