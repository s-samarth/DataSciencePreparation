# Segmentation & Detection

Classification answers "what is in this image?" Segmentation answers "which pixels belong to what?" and detection answers "where are the objects and what are they?" Both need spatial output that plain classifiers throw away.

!!! tip "Rapid Recall"
    Segmentation labels every pixel (semantic = by class, instance = per object, panoptic = both) with an encoder-decoder plus skip connections (U-Net): the encoder captures *what*, the decoder recovers *where*, and skips carry sharp detail across. Loss is cross-entropy + Dice (Dice fixes class imbalance); the metric is mIoU. Detection outputs boxes for an unknown number of objects: two-stage (Faster R-CNN, accurate) proposes then classifies, one-stage (YOLO) regresses boxes over a grid in one pass using anchors, with NMS to remove duplicates; scored by mAP@[0.5:0.95].

## §1 Image segmentation

Classification answers "what is in this image?" (one label). Segmentation answers **"which pixels belong to what?"**, a label for *every* pixel. The output is the same H×W as the input, each pixel holding a class instead of a color.

| Flavor | What it does |
| --- | --- |
| **Semantic** | Label every pixel by class. All cats are "cat"; two overlapping cats become one blob. No notion of individual objects. |
| **Instance** | Separate each *object*. Cat #1 and #2 get distinct masks. Counts "things," ignores "stuff" (sky, road). |
| **Panoptic** | The union: every pixel gets a class *and* countable objects get distinct instance IDs. Stuff + things. |

### Architecture, encoder-decoder

A classifier shrinks spatially and discards location to produce one label. Segmentation needs full-resolution output, so it uses an **encoder-decoder** (fully convolutional network):

- **Encoder (downsampling)**, a normal CNN. Shrinks spatially, grows channels. Captures *what* is present, loses precise *where*.
- **Decoder (upsampling)**, progressively upsamples back to full resolution via transposed convolutions or interpolation. Recovers *where*.
- **Skip connections (U-Net's key trick)**, copy high-resolution encoder feature maps across to the matching decoder level. Encoder-deep layers know *what*; encoder-early layers know precise *where* (sharp edges). Skips give the decoder both. Without them, boundaries are blurry.

<figure class="diagram diagram-dark" markdown="0">
<svg viewBox="0 0 680 300" xmlns="http://www.w3.org/2000/svg">
  <style>
    .e{fill:rgba(94,168,196,.18);stroke:#5ea8c4;stroke-width:1.8}
    .d{fill:rgba(224,179,65,.16);stroke:#e0b341;stroke-width:1.8}
    .t{fill:#b6b3aa;font-family:'Spline Sans Mono',monospace;font-size:11px;text-anchor:middle}
    .sk{stroke:#7fae6f;stroke-width:2;fill:none;stroke-dasharray:5 4;marker-end:url(#a3)}
    .fl{stroke:#8a8778;stroke-width:1.6;fill:none;marker-end:url(#a3)}
  </style>
  <defs><marker id="a3" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#7fae6f"/></marker></defs>
  <rect x="40"  y="40"  width="44" height="80" class="e"/><text x="62" y="135" class="t">enc1</text>
  <rect x="110" y="60"  width="40" height="56" class="e"/><text x="130" y="131" class="t">enc2</text>
  <rect x="176" y="80"  width="36" height="36" class="e"/><text x="194" y="131" class="t">enc3</text>
  <rect x="246" y="120" width="60" height="30" class="e" style="fill:rgba(201,122,90,.2);stroke:#c97a5a"/><text x="276" y="165" class="t">bottleneck</text>
  <rect x="350" y="80"  width="36" height="36" class="d"/><text x="368" y="131" class="t">dec3</text>
  <rect x="416" y="60"  width="40" height="56" class="d"/><text x="436" y="131" class="t">dec2</text>
  <rect x="486" y="40"  width="44" height="80" class="d"/><text x="508" y="135" class="t">dec1</text>
  <path d="M84 80 L110 85" class="fl"/><path d="M150 88 L176 95" class="fl"/><path d="M212 98 L246 130" class="fl"/>
  <path d="M306 135 L350 100" class="fl"/><path d="M386 95 L416 90" class="fl"/><path d="M456 85 L486 80" class="fl"/>
  <text x="20" y="84" class="t">in</text><path d="M28 80 L40 80" class="fl"/>
  <text x="560" y="84" class="t">mask</text><path d="M530 80 L548 80" class="fl"/>
  <path d="M84 50 Q 300 10 508 50" class="sk"/>
  <path d="M150 66 Q 290 30 436 66" class="sk"/>
  <path d="M212 84 Q 282 55 368 84" class="sk"/>
  <text x="300" y="20" class="t" style="fill:#7fae6f">skip connections (preserve sharp spatial detail)</text>
</svg>
<figcaption>U-Net: encoder shrinks, decoder grows, skip connections (green) carry high-resolution detail across so boundaries stay sharp.</figcaption>
</figure>

A normal conv maps a region to one value (downsample). A **transposed conv** reverses it, mapping one value to a region, learnably. Output size is $O = (W - 1)S - 2P + K$. People increasingly prefer **bilinear upsample + a 3×3 conv** instead, because transposed convs cause **checkerboard artifacts** (uneven kernel overlap when K isn't divisible by S).

### Getting data

Labels are **per-pixel masks**, an image where each pixel's value is its class ID. Expensive: tracing object boundaries can take 10 to 30 min/image. Formats: single-channel mask (pixel value = class index); instance segmentation also stores per-instance polygons / RLE (COCO). Datasets: COCO, Cityscapes, Pascal VOC, ADE20K. Cost mitigation: SAM (Segment Anything) generates masks from a click/box; weak supervision; augmentation.

!!! warning "Critical augmentation rule"
    Any **geometric** transform applied to the image must be applied *identically* to the mask (flip image → flip mask). **Photometric** transforms (brightness, blur) apply to the image only.

### Loss functions

Output: per-pixel logits $[B, C, H, W]$; target: $[B, H, W]$ of class indices. Two families, usually combined. Pixel-wise cross-entropy:

$$
\mathcal{L}_{CE} = -\frac{1}{HW}\sum_{p}\sum_{c} y_{p,c}\log \hat{y}_{p,c}
$$

Its problem is **class imbalance**: if 95% of pixels are background, CE is dominated by easy background and the model predicts "background everywhere." Fixes: class-weighted CE, or [focal loss](../losses/classification-losses.md) (down-weights easy pixels). Dice loss directly optimizes overlap:

$$
\text{Dice} = \frac{2\sum_p \hat{y}_p\, y_p}{\sum_p \hat{y}_p + \sum_p y_p}, \qquad \mathcal{L}_{Dice} = 1 - \text{Dice}
$$

It measures predicted ∩ truth over their sizes, so a tiny foreground region contributes equally regardless of surrounding background, and imbalance stops mattering. The **workhorse combo is $\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$**, especially in medical imaging.

### Metric, IoU

$$
\text{IoU} = \frac{|\text{pred} \cap \text{truth}|}{|\text{pred} \cup \text{truth}|} = \frac{TP}{TP + FP + FN}
$$

Compute per class, average to get **mIoU**, the headline number for semantic segmentation. Dice/F1 is also reported (medical); $\text{Dice} = 2\cdot\text{IoU}/(1+\text{IoU})$, so they are monotonic.

| Net | Idea |
| --- | --- |
| **FCN (2015)** | First fully-convolutional; replaced dense layers with convs, learned upsampling. |
| **U-Net (2015)** | Encoder-decoder with skip connections; dominant in medical/small-data. |
| **DeepLab v3+** | Atrous/dilated convolutions (enlarge receptive field without losing resolution) + ASPP (multi-scale). |
| **Mask R-CNN (2017)** | Instance-segmentation standard: detection + a mask head. |

## §2 Object detection and YOLO

Detection = **classification + localization, for multiple objects at once.** The output is a set of **bounding boxes**, each $(x, y, w, h, \text{class}, \text{confidence})$. It is harder than classification because the *number* of objects is unknown and variable, so there is no fixed-size output.

The central dichotomy is two paradigms:

| Two-stage (R-CNN family) | One-stage (YOLO, SSD, RetinaNet) |
| --- | --- |
| Accuracy-first. **Stage 1:** a Region Proposal Network proposes ~1000s of candidate regions. **Stage 2:** a CNN classifies + refines each. Lineage: R-CNN → Fast R-CNN (run CNN once, RoI-pool per region) → **Faster R-CNN** (learnable RPN, end-to-end) → **Mask R-CNN** (+ mask head = instance segmentation). | Speed-first. Skip proposals. In a **single forward pass**, directly predict boxes + classes over a grid. Real-time; historically slightly weaker on small/crowded objects, gap now largely closed. |

### How YOLO works ("You Only Look Once")

Frame detection as a **single regression problem over a grid**:

- Divide the image into an $S \times S$ grid (for example 13×13).
- Each cell predicts $B$ boxes $(x, y, w, h, \text{objectness})$ plus class probabilities. Objectness = "is there an object whose center falls in my cell, and how confident am I."
- One forward pass produces all boxes for the whole image at once, vs R-CNN looking thousands of times.

```
Image → 13×13 grid. The cell containing the object's CENTER detects it.
┌──┬──┬──┬──┐
│  │  │  │  │   Each cell outputs B boxes:
├──┼──┼──┼──┤     (x, y, w, h, conf) × B  +  class probs
│  │ ●│  │  │   ● = object center → that cell predicts the box
├──┼──┼──┼──┤
│  │  │  │  │
└──┴──┴──┴──┘
```

Predicting raw box sizes is unstable. Define a few **anchor boxes**, prior shapes (tall, wide, square) per cell, learned from dataset box statistics (k-means on training boxes). The network predicts *offsets/scales relative to anchors*, which is much easier to learn. (YOLO v8+ moved to **anchor-free**, predicting center+size directly, but anchors are the classic concept.)

!!! note "Interview note: Non-Max Suppression (NMS)"
    The grid produces many overlapping boxes for one object. NMS cleans up: sort boxes by confidence; keep the highest-confidence box; remove every other box whose **IoU with it exceeds a threshold** (for example 0.5), the duplicates of the same object; repeat with the next-highest remaining box. Without NMS you get 5 boxes around one cat. A near-guaranteed interview question.

The YOLO loss is multi-task:

$$
\mathcal{L} = \lambda_{coord}\mathcal{L}_{box} + \mathcal{L}_{obj} + \mathcal{L}_{cls}
$$

The **box loss** is regression on coordinates. Original YOLO used squared error on $(x, y, \sqrt{w}, \sqrt{h})$, the square root makes a 5px error matter more on a small box than a large one; modern variants use **IoU-based** losses (GIoU/DIoU/CIoU). The **objectness loss** asks does this box contain an object (BCE); $\lambda_{coord} \approx 5$ up-weights localization and $\lambda_{noobj} \approx 0.5$ down-weights the flood of empty boxes (the foreground/background imbalance that also motivated focal loss in RetinaNet). The **classification loss** is which class given an object (CE / BCE).

### Data and metric

Labels: bounding boxes $(\text{class}, x_c, y_c, w, h)$, usually normalized to [0,1]. Cheaper than masks, pricier than classification. Datasets: COCO (80 classes), Pascal VOC, Open Images. Augmentation: box-aware (flips/crops transform the boxes too) plus **Mosaic** (stitch 4 images, a YOLO signature).

For **mAP (mean Average Precision)**: a prediction is a True Positive if IoU with a ground-truth box ≥ threshold (for example 0.5) *and* the class matches. Sweep the confidence threshold to get a precision-recall curve; **Average Precision** is the area under it, per class; average AP over classes to get **mAP**. **mAP@0.5** uses IoU 0.5; **mAP@[0.5:0.95]** (COCO's primary) averages over IoU 0.5 to 0.95 in 0.05 steps, rewarding tight boxes.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")          # nano, pretrained on COCO
results = model("image.jpg")        # boxes + classes + conf
results[0].boxes.xyxy               # box coords
model.train(data="my_data.yaml", epochs=50, imgsz=640)  # fine-tune
```
