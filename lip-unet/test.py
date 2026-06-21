"""
test.py

Load the best saved model and run it on the test set.
For n_size samples, display the original image, ground truth mask, and predicted mask side by side.
Also prints the dice score for the batch.
"""

import torch
import matplotlib.pyplot as plt
from losses import calculate_dice_score
from model import build_model, device
from dataset import test_loader


def test(test_loader, model, n_size, device):
    # grab one batch from the test loader
    images, masks = next(iter(test_loader))
    images = images.to(device)
    masks = masks.to(device)
    model = model.to(device)

    model.eval()
    # no gradient calculation needed during inference
    with torch.no_grad():
        output = model(images)
        # threshold sigmoid output at 0.5 to get binary prediction mask
        pred_mask = (torch.sigmoid(output) > 0.5).float()

    dice_score = calculate_dice_score(output, masks)
    print(f"Dice score on this batch: {dice_score:.4f}")

    fig, axes = plt.subplots(n_size, 3, figsize=(10, 4 * n_size))

    for idx in range(n_size):
        # permute because matplotlib expects [H, W, C] but torch tensor is [C, H, W]
        img = images[idx].cpu().permute(1, 2, 0).numpy()
        # send mask to cpu and convert to numpy for plotting
        true_mask = masks[idx, 0].cpu().numpy()
        pred = pred_mask[idx, 0].cpu().numpy()

        axes[idx, 0].imshow(img)
        axes[idx, 0].set_title("original image")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(true_mask, cmap="gray")
        axes[idx, 1].set_title("ground truth mask")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(pred, cmap="gray")
        axes[idx, 2].set_title("predicted mask")
        axes[idx, 2].axis("off")

    plt.tight_layout()
    plt.show()


model = build_model()
# load best saved weights into the model
model.load_state_dict(torch.load("best.pth", map_location=device))
test(test_loader, model, 5, device)