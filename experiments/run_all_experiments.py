#!/usr/bin/env python3
"""
Comprehensive training script for all three underwater object detection approaches.
This script will train each approach and generate comparative results.
"""

import argparse
import os
import sys
import time
import json
from typing import Dict, Any
import logging
import tensorflow as tf
from datetime import datetime

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'approach1_yolov8'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'approach2_faster_rcnn'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'approach3_detr'))

from common_utils import setup_gpu, load_dataset, create_visualization_callbacks
from evaluation_utils import UnderwaterEvaluator, ComparativeAnalyzer

# Import approach-specific trainers
YOLOv8Trainer = None
FasterRCNNTrainer = None
DETRTrainer = None

try:
    from yolov8_underwater import YOLOv8Trainer
except ImportError as e:
    print(f"Warning: Could not import YOLOv8Trainer: {e}")

try:
    from faster_rcnn_underwater import FasterRCNNTrainer
except ImportError as e:
    print(f"Warning: Could not import FasterRCNNTrainer: {e}")

try:
    from detr_underwater import DETRTrainer
except ImportError as e:
    print(f"Warning: Could not import DETRTrainer: {e}")

if not any([YOLOv8Trainer, FasterRCNNTrainer, DETRTrainer]):
    print("Error: No trainers could be imported. Check dependencies.")
    print("Try installing missing packages: pip install wandb")

def setup_logging(experiment_dir: str) -> logging.Logger:
    """Setup comprehensive logging for the training experiment"""
    
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('underwater_training')
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_experiment_config(approach: str) -> Dict[str, Any]:
    """Load configuration for specific approach"""
    
    config_path = os.path.join('configs', f'{approach}_config.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}. Using default settings.")
        return get_default_config(approach)
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)
        
        # Flatten the nested config structure
        flat_config = {}
        
        # Extract nested values and flatten them
        if 'training' in raw_config:
            flat_config.update(raw_config['training'])
        if 'dataset' in raw_config:
            flat_config.update(raw_config['dataset'])
        if 'model' in raw_config:
            flat_config.update(raw_config['model'])
        
        # Add top-level keys
        for key, value in raw_config.items():
            if not isinstance(value, dict):
                flat_config[key] = value
        
        # Ensure required keys exist with defaults
        defaults = get_default_config(approach)
        for key, default_value in defaults.items():
            if key not in flat_config:
                flat_config[key] = default_value
        
        return flat_config
    except ImportError:
        print("PyYAML not installed. Using default configuration.")
        return get_default_config(approach)
    except Exception as e:
        print(f"Error loading config: {e}. Using default settings.")
        return get_default_config(approach)

def get_default_config(approach: str) -> Dict[str, Any]:
    """Get default configuration for approach"""
    
    base_config = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001,
        'image_size': 640,
        'num_classes': 7,
        'class_names': ['Fish', 'Jellyfish', 'Penguin', 'Puffin', 'Shark', 'Starfish', 'Stingray'],
        'early_stopping_patience': 15,
        'reduce_lr_patience': 10,
        'save_best_only': True,
        'validation_split': 0.2
    }
    
    # Approach-specific modifications
    if approach == 'yolov8':
        base_config.update({
            'confidence_threshold': 0.25,
            'nms_threshold': 0.45,
            'anchor_sizes': [16, 32, 64],
            'aspect_ratios': [0.5, 1.0, 2.0]
        })
    elif approach == 'faster_rcnn':
        base_config.update({
            'backbone': 'resnet50',
            'fpn_channels': 256,
            'rpn_batch_size': 256,
            'roi_batch_size': 512,
            'nms_threshold': 0.7
        })
    elif approach == 'detr':
        base_config.update({
            'hidden_dim': 256,
            'num_heads': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dropout': 0.1,
            'num_queries': 100
        })
    
    return base_config

def train_single_approach(approach: str, config: Dict[str, Any], 
                         dataset_path: str, experiment_dir: str, 
                         logger: logging.Logger) -> Dict[str, Any]:
    """Train a single approach and return results"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training for {approach.upper()} approach")
    logger.info(f"{'='*60}")
    
    approach_dir = os.path.join(experiment_dir, f'approach_{approach}')
    os.makedirs(approach_dir, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Setup GPU
        setup_gpu()
        
        # Load dataset
        logger.info("Loading dataset...")
        train_dataset, val_dataset, test_dataset = load_dataset(
            dataset_path, 
            batch_size=config['batch_size'],
            image_size=config['image_size'],
            validation_split=config['validation_split']
        )
        
        # Create trainer based on approach
        trainer = None
        if approach == 'yolov8' and YOLOv8Trainer is not None:
            trainer = YOLOv8Trainer(config)
        elif approach == 'faster_rcnn' and FasterRCNNTrainer is not None:
            trainer = FasterRCNNTrainer(config)
        elif approach == 'detr' and DETRTrainer is not None:
            trainer = DETRTrainer(config)
        else:
            raise ValueError(f"Trainer not available for approach: {approach}. Check dependencies.")
        
        # Setup callbacks
        callbacks = create_visualization_callbacks(
            model_dir=approach_dir,
            patience=config.get('early_stopping_patience', 15),
            reduce_lr_patience=config.get('reduce_lr_patience', 10),
            save_best_only=config.get('save_best_only', True)
        )
        
        # Train model
        logger.info(f"Training {approach} model...")
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=config['epochs'],
            callbacks=callbacks
        )
        
        # Evaluate model
        logger.info(f"Evaluating {approach} model...")
        test_results = trainer.evaluate(test_dataset)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(approach_dir, f'{approach}_best_model.h5')
        trainer.save_model(model_path)
        
        # Generate predictions for detailed evaluation
        logger.info("Generating predictions for detailed evaluation...")
        predictions = trainer.predict(test_dataset)
        
        # Save results
        results = {
            'approach': approach,
            'config': config,
            'training_time': training_time,
            'epochs_trained': len(history.history.get('loss', [])),
            'best_val_loss': min(history.history.get('val_loss', [float('inf')])),
            'test_results': test_results,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        results_path = os.path.join(approach_dir, 'training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save training history
        history_path = os.path.join(approach_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f, indent=2, default=str)
        
        logger.info(f"{approach.upper()} training completed successfully!")
        logger.info(f"   Training time: {training_time:.2f} seconds")
        logger.info(f"   Best validation loss: {results['best_val_loss']:.4f}")
        logger.info(f"   Results saved to: {results_path}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error training {approach}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'approach': approach,
            'error': str(e),
            'training_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }

def run_comparative_evaluation(experiment_dir: str, logger: logging.Logger) -> None:
    """Run comprehensive comparative evaluation across all approaches"""
    
    logger.info(f"\n{'='*60}")
    logger.info("Starting Comparative Evaluation")
    logger.info(f"{'='*60}")
    
    try:
        # Class names for underwater dataset
        class_names = ['Fish', 'Jellyfish', 'Penguin', 'Puffin', 'Shark', 'Starfish', 'Stingray']
        
        # Create evaluator and analyzer
        evaluator = UnderwaterEvaluator(class_names)
        analyzer = ComparativeAnalyzer(class_names)
        
        # Load results from each approach
        approaches = ['yolov8', 'faster_rcnn', 'detr']
        
        for approach in approaches:
            results_path = os.path.join(experiment_dir, f'approach_{approach}', 'training_results.json')
            
            if os.path.exists(results_path):
                logger.info(f"Loading results for {approach}...")
                analyzer.load_results(approach, results_path)
            else:
                logger.warning(f"Results not found for {approach}: {results_path}")
        
        # Generate comparative report
        logger.info("Generating comparative analysis report...")
        report = analyzer.generate_comparative_report()
        
        # Save report
        report_path = os.path.join(experiment_dir, 'comparative_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Generate visualization plots
        logger.info("Generating comparative visualization plots...")
        plots_dir = os.path.join(experiment_dir, 'comparative_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        analyzer.plot_comparative_results(plots_dir)
        
        logger.info(f"Comparative evaluation completed!")
        logger.info(f"   Report saved to: {report_path}")
        logger.info(f"   Plots saved to: {plots_dir}")
        
    except Exception as e:
        logger.error(f"Error in comparative evaluation: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train underwater object detection models')
    parser.add_argument('--approach', type=str, choices=['yolov8', 'faster_rcnn', 'detr', 'all'],
                       default='all', help='Approach to train (default: all)')
    parser.add_argument('--dataset', type=str, 
                       default='../preprocessed_dataset',
                       help='Path to preprocessed dataset directory')
    parser.add_argument('--experiment_dir', type=str,
                       default='./experiments_results',
                       help='Directory to save experiment results')
    parser.add_argument('--config_dir', type=str,
                       default='./configs',
                       help='Directory containing configuration files')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and run only comparative evaluation')
    
    args = parser.parse_args()
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.experiment_dir, f'experiment_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir)
    
    logger.info("Underwater Object Detection Training Framework")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Dataset path: {args.dataset}")
    logger.info(f"Approaches to train: {args.approach}")
    
    # Check dataset availability
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found: {args.dataset}")
        logger.error("Please ensure the preprocessed dataset is available.")
        return 1
    
    # Determine which approaches to train
    if args.approach == 'all':
        approaches = ['yolov8', 'faster_rcnn', 'detr']
    else:
        approaches = [args.approach]
    
    # Training phase
    if not args.skip_training:
        training_results = {}
        
        for approach in approaches:
            try:
                # Load configuration
                config = load_experiment_config(approach)
                
                # Train approach
                results = train_single_approach(
                    approach=approach,
                    config=config,
                    dataset_path=args.dataset,
                    experiment_dir=experiment_dir,
                    logger=logger
                )
                
                training_results[approach] = results
                
                # Save intermediate results
                interim_results_path = os.path.join(experiment_dir, 'interim_training_results.json')
                with open(interim_results_path, 'w') as f:
                    json.dump(training_results, f, indent=2, default=str)
                
            except KeyboardInterrupt:
                logger.info("\nTraining interrupted by user.")
                logger.info("Partial results saved. You can run comparative evaluation separately.")
                break
            except Exception as e:
                logger.error(f"Unexpected error training {approach}: {e}")
                continue
        
        # Save final training results
        final_results_path = os.path.join(experiment_dir, 'all_training_results.json')
        with open(final_results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"\nTraining Summary:")
        for approach, result in training_results.items():
            if 'error' in result:
                logger.info(f"   {approach.upper()}: Failed - {result['error']}")
            else:
                logger.info(f"   {approach.upper()}: Completed in {result['training_time']:.1f}s")
    
    # Comparative evaluation phase
    if len(approaches) > 1 or args.skip_training:
        try:
            run_comparative_evaluation(experiment_dir, logger)
        except Exception as e:
            logger.error(f"Comparative evaluation failed: {e}")
    
    logger.info(f"\nExperiment completed! Results saved to: {experiment_dir}")
    return 0

if __name__ == "__main__":
    exit(main())
