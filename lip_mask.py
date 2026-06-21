"""
In this file I apply color on the lips using the predicted mask:
> run the image through the onnx model to get the lip mask
> create a solid color layer (same size as image) filled with the chosen lip color
> blend the original image with the color layer using addWeighted (0.6 original, 0.4 color)
> stack the mask from [256,256] to [256,256,3] so it matches the image shape
> use np.where to apply the blended color only where mask=1 (lip pixels), keep original everywhere else
"""

import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# load the onnx model 
session = ort.InferenceSession("model.onnx")

# load and preprocess your image 
image = cv2.imread("download.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (256, 256))

# normalize same way as training (mean=0, std=1, max=255)
image_norm = image.astype(np.float32) / 255.0

# [H, W, C] → [1, C, H, W]
image_input = np.transpose(image_norm, (2, 0, 1))
image_input = np.expand_dims(image_input, axis=0)

# run inference
outputs = session.run(["output"], {"input": image_input})
logits = outputs[0]

pred_mask = 1 / (1 + np.exp(-logits))
pred_mask = (pred_mask > 0.5).astype(np.uint8)
pred_mask = pred_mask[0, 0]  # [256, 256]

#apply color overlay on lips 
color = (255, 0, 100)  # RGB color you want on lips

# create a solid color image same size as original
color_layer = np.zeros_like(image)  # [256, 256, 3]
color_layer[:] = color              # fill entire layer with lip color



# blend original image with color layer only where mask=1
lip_overlay = cv2.addWeighted(image, 0.6, color_layer, 0.4, 0)

lip_mask = np.stack([pred_mask, pred_mask, pred_mask], axis=-1)  # [256,256,3]
lip_result = np.where(lip_mask == 1, lip_overlay, image)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image)
axes[0].set_title("original image")
axes[0].axis("off")

axes[1].imshow(pred_mask, cmap="gray")
axes[1].set_title("predicted mask")
axes[1].axis("off")

axes[2].imshow(lip_result)
axes[2].set_title("lip color overlay")  # FIX: was imshow result, not a string title
axes[2].axis("off")

plt.tight_layout()
plt.show()
