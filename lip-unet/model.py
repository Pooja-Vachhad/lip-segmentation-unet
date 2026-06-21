"""
model.py 

Here we are using pre-trained weights from resnet18 and fine-tuning to our dataset to learn 
from our dataset features. 
Using checkpoint save/load to make sure the best loss is saved 
and training can resume from where it stopped.
"""

import os
import torch 
import segmentation_models_pytorch as smp


checkpoint_path = "checkpoint.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1, 
        activation=None, # no activation here because BCEWithLogitsLoss expects raw logits
    )
    return model.to(device)


def save_checkpoint(model, optimizer, epoch, best_loss, counter, history, path=checkpoint_path):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
        "counter": counter,
        "history": history,
    }, path)


def load_checkpoint(model, optimizer, path=checkpoint_path):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"resuming from epoch {checkpoint['epoch']}")
        return (
            checkpoint["epoch"],
            checkpoint["best_loss"],
            checkpoint.get("counter", 0),
            checkpoint.get("history", {
                "train_loss": [], "valid_loss": [],
                "train_dice": [], "valid_dice": []
            })
        )
    print("no checkpoint found, starting fresh")
    return 0, float("inf"), 0, {
        "train_loss": [], "valid_loss": [],
        "train_dice": [], "valid_dice": []
    }