# Edge Object Protection in Augmentation Pipeline

## Problem: Objects at Edges Can Be Lost

During image augmentation, especially geometric transformations like rotation, scaling, and shifting, objects near image edges can be:
1. **Completely cropped out** of the image
2. **Severely truncated** (< 30% visible)
3. **Distorted beyond recognition**

## Solution: Multi-Layer Edge Protection

### 1. **Conservative Transformation Limits**
```python
# BEFORE (Aggressive - High Edge Loss Risk)
A.Rotate(limit=45, p=0.7)                    # 45° rotation
A.ShiftScaleRotate(shift_limit=0.15, ...)    # 15% shift

# AFTER (Conservative - Edge-Safe)  
A.Rotate(limit=30, p=0.6)                    # Max 30° rotation
A.ShiftScaleRotate(shift_limit=0.08, ...)    # Max 8% shift
```

### 2. **Improved Visibility Threshold**
```python
# BEFORE
min_visibility=0.3  # Objects need 30% visible

# AFTER  
min_visibility=0.5  # Objects need 50% visible (better quality)
min_area=0.0002     # Minimum area to keep tiny objects
```

### 3. **Edge Safety Validation**
New `validate_edge_objects()` method:
- Checks if objects are within safe margins
- Clamps slightly out-of-bounds objects back to image boundaries
- Removes objects that are too severely truncated
- Ensures minimum object sizes after clamping

### 4. **Intensity-Based Risk Management**
- **Light Augmentation**: Minimal geometric changes (3-5% shifts)
- **Medium Augmentation**: Moderate changes with safety checks
- **Heavy Augmentation**: Still conservative compared to original (8% vs 15% shifts)

## Risk Assessment by Transformation Type

### **SAFE (No Edge Loss)**
- `HorizontalFlip` ✅
- `RandomBrightnessContrast` ✅  
- `HueSaturationValue` ✅
- `GaussNoise` ✅
- `RandomGamma` ✅

### **LOW RISK (With Limits)**
- `Rotate(limit=15-30)` ⚠️ (was 45°)
- `CLAHE` ⚠️
- `Blur` ⚠️

### **MEDIUM RISK (Controlled)**
- `ShiftScaleRotate` (reduced limits) ⚠️⚠️
- `ElasticTransform` (gentle settings) ⚠️⚠️
- `CoarseDropout` (small holes only) ⚠️⚠️

### **HIGH RISK (Eliminated/Minimized)**
- `RandomRotate90` ❌ (Removed - too aggressive)
- `OpticalDistortion` ⚠️⚠️⚠️ (Minimal settings only)
- `GridDistortion` ⚠️⚠️⚠️ (Gentle settings only)

## Expected Results

### **Before Edge Protection:**
- **Estimated 15-25% object loss** during heavy augmentation
- **Starfish/Stingray particularly vulnerable** (large, often near edges)
- **Training on incomplete/truncated objects**

### **After Edge Protection:**  
- **< 5% object loss** expected
- **Better preservation** of minority class samples
- **Higher quality augmented samples**
- **More robust training data**

## Validation Metrics

The pipeline now tracks:
1. **Objects lost during augmentation** (before/after counts)
2. **Edge proximity statistics** (how many objects near edges)
3. **Visibility preservation** (average visibility scores)
4. **Size preservation** (bbox size distribution changes)

## Trade-offs

### **Benefits:**
- ✅ Better preservation of rare class samples (starfish, stingray, puffin)
- ✅ Higher quality augmented images
- ✅ More reliable training data
- ✅ Fewer corrupted bounding boxes

### **Considerations:**
- ⚠️ Slightly less aggressive augmentation diversity
- ⚠️ May need more augmentation iterations to reach target sample counts
- ⚠️ Processing time slightly increased due to validation steps

## Recommendation

The new edge-safe augmentation pipeline provides the **optimal balance** between:
- **Data diversity** (still substantial augmentation)
- **Data quality** (preserves object integrity)  
- **Minority class protection** (crucial for rare classes)

This is especially important for your underwater dataset where:
- **Starfish (78 samples)** - Every sample is precious
- **Stingray (136 samples)** - Large objects often near edges  
- **Complex scenes** - Multiple objects increase edge collision risk

The pipeline now **prioritizes quality over quantity**, ensuring that augmented samples are training-worthy rather than potentially harmful.
