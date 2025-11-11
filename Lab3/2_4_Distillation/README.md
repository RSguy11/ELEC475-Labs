# Knowledge Distillation Lab: Response vs Feature-Based Methods

This lab implements two distinct knowledge distillation approaches for semantic segmentation:
1. **Response-Based Distillation** - Using output predictions with temperature scaling
2. **Feature-Based Distillation** - Using intermediate feature maps with cosine similarity

## 📁 Directory Structure

```
Lab3/2_4_Distillation/
├── response_based_distillation/
│   ├── train.py                    # Pure response-based training
│   ├── test.py                     # Evaluation and visualization
│   ├── results_images/             # Training losses and prediction plots
│   └── response_based_model.pth    # Best trained model
└── feature_based_distillation/
    ├── train.py                    # Pure feature-based training
    ├── test.py                     # Evaluation and visualization
    ├── results_images/             # Training losses and prediction plots
    └── feature_based_model.pth     # Best trained model
```

## 🎯 Method Comparison

### Response-Based Distillation
- **Method**: Knowledge Distillation with Temperature Scaling
- **Loss**: KL Divergence between soft predictions
- **Temperature**: τ = 4.0 
- **Weights**: 70% soft targets + 30% hard targets (standard KD weights)
- **Focus**: Learning from teacher's prediction confidence/uncertainty

### Feature-Based Distillation  
- **Method**: Intermediate Feature Matching
- **Loss**: Cosine Similarity Loss on feature maps
- **Layers**: Match at 3 different encoder depths
- **Weights**: 70% feature matching + 30% hard targets (standard FD weights)
- **Focus**: Learning from teacher's internal representations

## 🚀 Training Instructions

### Response-Based Training
```bash
cd response_based_distillation
python train.py
```

### Feature-Based Training
```bash
cd feature_based_distillation  
python train.py
```

Both training scripts will:
- **Start from random initialization** (no pre-trained student model)
- Train for 30 epochs using pure knowledge distillation
- Save best model based on validation mIoU
- Generate loss plots in `results_images/`
- Print training progress and metrics
- Demonstrate effectiveness of each distillation method from scratch

## 📊 Evaluation Instructions

### Response-Based Evaluation
```bash
cd response_based_distillation
python test.py
```

### Feature-Based Evaluation
```bash
cd feature_based_distillation
python test.py
```

Each evaluation script generates:
- **mIoU metrics** for teacher vs student models
- **Per-class IoU scores** for detailed analysis
- **Prediction comparisons** visual plots
- **Performance summary** charts
- **Knowledge transfer analysis**

## 🔬 Expected Results

### Training Output
Each method produces separate:
- Training loss curves
- Validation mIoU progression  
- Best model checkpoints
- Method-specific loss components

### Evaluation Output
Comparative analysis including:
- Teacher mIoU vs Student mIoU
- Knowledge transfer percentage
- Visual prediction quality
- Feature similarity (for feature-based method)

## 📈 Key Differences

| Aspect | Response-Based | Feature-Based |
|--------|---------------|---------------|
| **Learning Target** | Output predictions | Internal features |
| **Loss Function** | KL Divergence | Cosine Similarity |
| **Temperature** | τ = 4.0 | N/A |
| **Adaptation** | None needed | Conv1x1 layers |
| **Loss Weights** | α=0.7, β=0.3 | γ=0.7, β=0.3 |
| **Complexity** | Lower | Higher |
| **Memory Usage** | Lower | Higher |

## 🎓 Educational Goals

This lab demonstrates:
1. **Response-based distillation** effectiveness from random initialization
2. **Feature-based distillation** for representation learning from scratch
3. **Comparative analysis** of both approaches without pre-training bias
4. **Pure knowledge transfer** techniques with equal starting conditions
5. **Evaluation methodologies** for distillation quality assessment

## 🔧 Technical Details

### Models
- **Teacher**: FCN-ResNet50 (35.3M parameters)
- **Student**: SMNet base16 (313K parameters)  
- **Compression**: ~112x parameter reduction

### Dataset
- **Dataset**: PASCAL VOC 2012 Segmentation
- **Classes**: 21 semantic classes
- **Input Size**: 224×224 pixels
- **Batch Size**: 8

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 30
- **Device**: CUDA (GPU required)
- **Loss Weights**: Optimized for each method

Run both methods to compare their effectiveness for knowledge distillation in semantic segmentation!