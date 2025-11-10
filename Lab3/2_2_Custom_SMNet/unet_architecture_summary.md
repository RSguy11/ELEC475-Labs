**Architecture Overview**

```
Image → [Backbone/Encoder: DW-Conv stages ↓↓] → Bottleneck → [Decoder: Upsample + Skip + Conv ↑↑↑] → 1×1 Head → Segmentation mask
              f1 (H/2)        f2 (H/4)        f3 (H/8)        f4 (H/16)
                          ↖──────────────┬──────────────↗
                                        Skips
```

 **Implementation**
