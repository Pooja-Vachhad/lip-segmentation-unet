# Lip Segmentation & Virtual Try-On

Apply lipstick shades virtually using a UNet segmentation model trained to detect lip regions with **98.12% Dice Score**.

<img width="990" height="1928" alt="Image" src="https://github.com/user-attachments/assets/a69da4ec-7b3e-4488-aaef-fd15b46c9726" />

**Live Demo:** _link coming soon_

---

## Project Overview

This project predicts a lip mask from a face image using a deep learning segmentation model, then overlays a chosen lipstick color only on the lip region. Built for the beauty/makeup industry so customers can try shades virtually no product wasted, no time lost.

---

<img width="1174" height="407" alt="Image" src="https://github.com/user-attachments/assets/46815e5a-4698-4759-8c39-389ba556bf34" />


## Training Results

| Metric | Value |
|---|---|
| Best Train Loss | 0.0351 |
| Best Valid Loss | 0.0269 |
| Dice Score (Test Set) | **0.9812** |
| Total Epochs Run | 50 |
| Early Stopping Triggered | No (model kept slowly improving) |

<img width="989" height="390" alt="Image" src="https://github.com/user-attachments/assets/5588d0e7-36e7-4e2c-9d91-ec5a48b4bb20" />

---

## Training Configuration

| Setting | Value | Why |
|---|---|---|
| Epochs | 50 | Model kept improving slowly, patience never maxed out |
| Batch Size | 32 | Tried 16 first — 32 gave better stability and faster convergence |
| Optimizer | Adam (lr=1e-4) | Works well out of the box for segmentation tasks |
| Loss Function | DiceBCE Loss | BCE alone ignores foreground (lips); Dice forces the model to focus on it |
| Scheduler | ReduceLROnPlateau | Cuts LR by half if valid loss doesn't improve for 3 epochs, giving model another chance |
| Scheduler Mode | min | Monitoring validation loss (lower = better) |
| Scheduler Factor | 0.5 | LR halved each time: 1e-4 → 5e-5 → 2.5e-5 |
| Scheduler Patience | 3 | Waits 3 epochs before reducing LR |
| Early Stopping Patience | 7 | Stops training if no improvement for 7 consecutive epochs |
| Checkpoint | Saved at best valid loss | So training can resume without losing progress |

---

## Model Architecture

| Component | Choice | Why |
|---|---|---|
| Architecture | UNet | Best for binary segmentation — encoder extracts features, decoder reconstructs mask |
| Encoder | ResNet-18 | Pre-trained on ImageNet, strong feature extractor out of the box |
| Encoder Weights | ImageNet | Model already knows edges, textures, shapes fine-tuned to lip dataset |
| Output Classes | 1 | Binary segmentation: lip (1) vs background (0) |
| Activation | None | Raw logits passed to BCEWithLogitsLoss which applies sigmoid internally |
| Export | ONNX (opset 11) | Lightweight inference without PyTorch dependency |

---

## Data Augmentation

| Augmentation | Value | Why |
|---|---|---|
| Resize | 256×256 | Consistent input size for the model |
| Horizontal Flip | p=0.5 | Increases diversity, lips are symmetric |
| Rotate | limit=35°, p=0.5 | Handles tilted face angles |
| Normalize | mean=0, std=1 | Scales pixels to 0–1 range |

---

## Dataset

[Lip Segmentation — Kaggle](https://www.kaggle.com/datasets/utkarshsaxenadn/lip-segmentation-attention-unet)

Split: **80% train / 10% validation / 10% test** with fixed seed (42) for reproducibility.

---

## Challenges & How I Overcame Them

| Challenge | What Happened | How I Fixed It |
|---|---|---|
| GPU memory crash mid-training | Colab disconnected around epoch 35 | Implemented checkpoint save/load so training resumed from best loss |
| Poor early performance | Loss was high with batch_size=16 | Switched to batch_size=32 — more stable gradients, faster convergence |
| LR getting stuck | Loss plateaued for several epochs | Added ReduceLROnPlateau scheduler to automatically reduce LR when stuck |

---

## Real World Use Case

In the **beauty and makeup industry**, customers often buy lipstick shades they end up not liking once applied. This tool lets them:

- Upload a photo
- Pick from 25 shades across Pink, Nude, Red, Blue, and Green families
- See the color applied on their lips instantly

No product wasted. No guessing. Try before you buy.

---

## Tech Stack

```
PyTorch · segmentation-models-pytorch · Albumentations
ONNX · ONNXRuntime · OpenCV · Streamlit
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
