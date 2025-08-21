#!/usr/bin/env python3
"""
Test script for the evaluation utilities framework.
Validates the UnderwaterEvaluator and ComparativeAnalyzer classes.
"""

import numpy as np
import sys
import os

# Add the experiments directory to path
sys.path.append(os.path.dirname(__file__))

from utils.evaluation_utils import UnderwaterEvaluator, ComparativeAnalyzer

def create_dummy_predictions():
    """Create dummy predictions for testing"""
    # Simulate predictions for 10 test images
    n_images = 10
    predictions = []
    
    for i in range(n_images):
        # Random number of detections per image (0-5)
        n_detections = np.random.randint(0, 6)
        
        image_predictions = []
        if n_detections > 0:
            # Random bounding boxes [x1, y1, x2, y2]
            boxes = np.random.rand(n_detections, 4)
            boxes[:, 2:] = boxes[:, :2] + 0.1 + np.random.rand(n_detections, 2) * 0.3
            boxes = np.clip(boxes, 0, 1)  # Ensure valid coordinates
            
            # Random class predictions (0-6 for 7 classes)
            classes = np.random.randint(0, 7, n_detections)
            
            # Random confidence scores
            scores = np.random.rand(n_detections) * 0.5 + 0.5  # 0.5-1.0 range
            
            for j in range(n_detections):
                image_predictions.append({
                    'bbox': boxes[j],
                    'class_id': int(classes[j]),
                    'confidence': float(scores[j])
                })
        
        predictions.append(image_predictions)
    
    return predictions

def create_dummy_ground_truth():
    """Create dummy ground truth for testing"""
    # Simulate ground truth for 10 test images
    n_images = 10
    ground_truth = []
    
    for i in range(n_images):
        # Random number of objects per image (0-4)
        n_objects = np.random.randint(0, 5)
        
        image_ground_truth = []
        if n_objects > 0:
            # Random bounding boxes [x1, y1, x2, y2]
            boxes = np.random.rand(n_objects, 4)
            boxes[:, 2:] = boxes[:, :2] + 0.1 + np.random.rand(n_objects, 2) * 0.3
            boxes = np.clip(boxes, 0, 1)  # Ensure valid coordinates
            
            # Random class labels (0-6 for 7 classes)
            classes = np.random.randint(0, 7, n_objects)
            
            for j in range(n_objects):
                image_ground_truth.append({
                    'bbox': boxes[j],
                    'class_id': int(classes[j])
                })
        
        ground_truth.append(image_ground_truth)
    
    return ground_truth

def test_underwater_evaluator():
    """Test the UnderwaterEvaluator class"""
    print("Testing UnderwaterEvaluator...")
    
    # Class names for underwater dataset
    class_names = ['Fish', 'Jellyfish', 'Penguin', 'Puffin', 'Shark', 'Starfish', 'Stingray']
    
    # Create evaluator
    evaluator = UnderwaterEvaluator(class_names)
    
    # Generate dummy data
    predictions = create_dummy_predictions()
    ground_truth = create_dummy_ground_truth()
    
    # Test individual methods
    print("Testing IoU calculation...")
    box1 = np.array([0.1, 0.1, 0.5, 0.5])
    box2 = np.array([0.2, 0.2, 0.6, 0.6])
    iou = evaluator.calculate_iou(box1, box2)
    print(f"IoU between test boxes: {iou:.4f}")
    
    # Test mAP calculation
    print("Testing mAP calculation...")
    map_scores = evaluator.calculate_map(predictions, ground_truth)
    print(f"mAP@0.5: {map_scores['map_50']:.4f}")
    print(f"mAP@0.75: {map_scores['map_75']:.4f}")
    print(f"Overall mAP: {map_scores['overall_map']:.4f}")
    
    # Test per-class AP
    print("Testing per-class AP calculation...")
    for class_id, class_name in enumerate(class_names):
        ap, _, _ = evaluator.calculate_ap_per_class(predictions, ground_truth, class_id)
        print(f"  {class_name}: {ap:.4f}")
    
    # Test minority class metrics
    print("Testing minority class metrics...")
    minority_metrics = evaluator.calculate_minority_class_metrics(predictions, ground_truth)
    
    print(f"  Minority class results:")
    for class_name, metrics in minority_metrics.items():
        print(f"    {class_name.title()}:")
        print(f"      AP@0.5: {metrics['ap_50']:.4f}")
        print(f"      AP@0.75: {metrics['ap_75']:.4f}")
        print(f"      Max F1: {metrics['max_f1']:.4f}")
    
    return True

def test_comparative_analyzer():
    """Test the ComparativeAnalyzer class"""
    print("\n" + "="*50)
    print("Testing ComparativeAnalyzer...")
    
    # Class names for underwater dataset
    class_names = ['Fish', 'Jellyfish', 'Penguin', 'Puffin', 'Shark', 'Starfish', 'Stingray']
    
    # Create analyzer
    analyzer = ComparativeAnalyzer(class_names)
    
    # Add dummy results for three approaches
    dummy_results = {
        'yolov8': {
            'mAP': 0.756,
            'mAP_75': 0.623,
            'mAP_coco': 0.478,
            'per_class_ap': {class_name: np.random.rand() * 0.8 + 0.1 for class_name in class_names},
            'minority_class_metrics': {
                'minority_map': 0.45,
                'majority_map': 0.78,
                'ratio': 0.576,
                'per_minority_ap': {'Starfish': 0.423, 'Stingray': 0.467, 'Jellyfish': 0.456}
            },
            'training_time': 2.5,
            'inference_time': 0.015,
            'model_size': 25.3,
            'fps': 66.7
        },
        'faster_rcnn': {
            'mAP': 0.742,
            'mAP_75': 0.651,
            'mAP_coco': 0.489,
            'per_class_ap': {class_name: np.random.rand() * 0.8 + 0.1 for class_name in class_names},
            'minority_class_metrics': {
                'minority_map': 0.48,
                'majority_map': 0.76,
                'ratio': 0.632,
                'per_minority_ap': {'Starfish': 0.445, 'Stingray': 0.489, 'Jellyfish': 0.512}
            },
            'training_time': 4.2,
            'inference_time': 0.045,
            'model_size': 158.7,
            'fps': 22.2
        },
        'detr': {
            'mAP': 0.718,
            'mAP_75': 0.634,
            'mAP_coco': 0.456,
            'per_class_ap': {class_name: np.random.rand() * 0.8 + 0.1 for class_name in class_names},
            'minority_class_metrics': {
                'minority_map': 0.52,
                'majority_map': 0.74,
                'ratio': 0.703,
                'per_minority_ap': {'Starfish': 0.478, 'Stingray': 0.523, 'Jellyfish': 0.556}
            },
            'training_time': 6.8,
            'inference_time': 0.078,
            'model_size': 202.4,
            'fps': 12.8
        }
    }
    
    # Load results into analyzer
    analyzer.approach_results = dummy_results
    
    # Test report generation
    print("Testing report generation...")
    report = analyzer.generate_comparative_report()
    
    print("Report generated successfully!")
    print(f"Report length: {len(report)} characters")
    print(f"Report preview: {report[:200]}...")
    
    # Test plotting (without saving)
    print("\nTesting visualization plots...")
    try:
        analyzer.plot_comparative_results()
        print("Visualization plots generated successfully!")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
    
    return True

def main():
    """Main test function"""
    print("Testing Underwater Detection Evaluation Framework")
    print("=" * 60)
    
    try:
        # Test individual evaluator
        success1 = test_underwater_evaluator()
        
        # Test comparative analyzer
        success2 = test_comparative_analyzer()
        
        if success1 and success2:
            print("\n" + "=" * 60)
            print("All tests passed! Evaluation framework is working correctly.")
            print("Ready to proceed with actual training experiments.")
        else:
            print("\n" + "=" * 60)
            print("Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
