"""
train.py

In this file I run the training loop for the model.
I save the loss and dice score for both train and validation at each epoch.
To calculate loss I use DiceBCELoss, and for the metric I use calculate_dice_score.
If the best validation loss doesn't improve for 7 consecutive epochs, early stopping kicks in
to save both training time and compute cost.
I use Adam optimizer as default since it works well with this model and dataset.
"""

import torch
import matplotlib.pyplot as plt
from losses import calculate_dice_score, DiceBCELoss
from tqdm import tqdm
from model import build_model, device, load_checkpoint, save_checkpoint
from dataset import train_loader, valid_loader

model = build_model()
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# reduce LR by half if validation loss doesn't improve for 3 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
checkpoint_path = "checkpoint.pth"
model_path = "best.pth"


def run_model(train_loader, valid_loader, epochs, device, criterion, optimizer, model, scheduler, output_path, patience):
    history = {"train_loss": [], "valid_loss": [], "train_dice": [], "valid_dice": []}
    best_loss = float("inf")
    counter = 0
    # load from checkpoint if one exists, otherwise start fresh
    start_epoch, best_loss, counter, history = load_checkpoint(model, optimizer)

    for epoch in tqdm(range(start_epoch, epochs)):
        # ── training phase ──
        model.train()
        train_loss = 0
        train_dice = 0

        for image, mask in train_loader:
            # move data to GPU/CPU
            image = image.to(device)
            mask = mask.to(device)

            output = model(image)
            loss = criterion(output, mask)

            optimizer.zero_grad()   # clear old gradients
            loss.backward()         # compute new gradients
            optimizer.step()        # update weights

            train_loss += loss.item()
            train_dice += calculate_dice_score(output, mask)

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        history["train_dice"].append(avg_train_dice)

        # ── validation phase ──
        model.eval()
        valid_loss = 0
        valid_dice = 0

        # no gradient computation during validation to save memory and speed things up
        with torch.no_grad():
            for image, mask in valid_loader:
                image = image.to(device)
                mask = mask.to(device)

                output = model(image)
                loss = criterion(output, mask)

                valid_loss += loss.item()
                valid_dice += calculate_dice_score(output, mask)

        avg_valid_loss = valid_loss / len(valid_loader)
        avg_valid_dice = valid_dice / len(valid_loader)

        # step scheduler after validation loss is computed
        scheduler.step(avg_valid_loss)

        history["valid_loss"].append(avg_valid_loss)
        history["valid_dice"].append(avg_valid_dice)

        # ── checkpoint + early stopping ──
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            # save best model weights separately
            torch.save(model.state_dict(), output_path)
            # save full checkpoint so training can resume
            save_checkpoint(model, optimizer, epoch, best_loss, counter, history)
            counter = 0
        else:
            counter += 1
            print(f"model not improving ({counter}/{patience})")

        print(f"epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}, valid_loss={avg_valid_loss:.4f} | train_dice={avg_train_dice:.4f}, valid_dice={avg_valid_dice:.4f}")

        if counter >= patience:
            print("patience exceeded, stopping early")
            break

    # ── plot loss and dice curves ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["valid_loss"], label="valid_loss")
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history["train_dice"], label="train_dice")
    axes[1].plot(history["valid_dice"], label="valid_dice")
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel("dice score")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return history


history = run_model(train_loader, valid_loader, 50, device, criterion, optimizer, model, scheduler, model_path, 7)
