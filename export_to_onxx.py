import torch
from mobiletunet import MobileUNetLite

device_str = "cpu"
ckpt_path = "./checkpoints/best_finetuned.pth"

model = MobileUNetLite().to(device_str)
ckpt = torch.load(ckpt_path, map_location=device_str)

model.load_state_dict(ckpt["state_dict"])

# create dummy input with the correct shape (batch, channels, H, W)
dummy = torch.randn(1, 3, 224, 224).to(device_str)


# export
torch.onnx.export(
    model,
    dummy,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    do_constant_folding=True
)
print("Saved model.onnx")
