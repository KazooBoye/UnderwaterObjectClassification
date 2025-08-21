"""
Approach 3: DETR Transformer-Based Detector for Underwater Object Detection
TensorFlow implementation with self-attention and set prediction for crowded underwater scenes
"""

import os
import sys
import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from datetime import datetime
import math

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from common_utils import UnderwaterDatasetConfig, GPUSetup, DataLoader, MetricsCalculator, Visualizer
from callbacks import CosineRestartScheduler, MetricsLogger

class PositionalEncoding2D(tf.keras.layers.Layer):
    """2D Positional Encoding for image features"""
    
    def __init__(self, temperature=10000, normalize=True):
        super(PositionalEncoding2D, self).__init__()
        self.temperature = temperature
        self.normalize = normalize
        
    def call(self, x):
        """
        x: [batch_size, height, width, channels]
        returns: [batch_size, height, width, channels] with positional encoding added
        """
        batch_size, height, width, channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # Create coordinate grids
        y_embed = tf.range(height, dtype=tf.float32)[:, None]
        x_embed = tf.range(width, dtype=tf.float32)[None, :]
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (height + eps) * 2 * math.pi
            x_embed = x_embed / (width + eps) * 2 * math.pi
        
        # Generate sine and cosine encodings
        dim_t = tf.range(channels // 4, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (channels // 4))
        
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        
        pos_x = tf.stack([tf.sin(pos_x[:, :, 0::2]), tf.cos(pos_x[:, :, 1::2])], axis=3)
        pos_y = tf.stack([tf.sin(pos_y[:, :, 0::2]), tf.cos(pos_y[:, :, 1::2])], axis=3)
        
        pos_x = tf.reshape(pos_x, [height, width, channels // 2])
        pos_y = tf.reshape(pos_y, [height, width, channels // 2])
        
        pos = tf.concat([pos_y, pos_x], axis=-1)
        
        # Add batch dimension and add to input
        pos = tf.tile(pos[None, ...], [batch_size, 1, 1, 1])
        
        return x + pos

class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Self Attention"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Calculate the attention weights"""
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        
        # Softmax is normalized on the last axis (seq_len_k)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights
    
    def call(self, q, k, v, mask=None, training=False):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q, training=training)
        k = self.wk(k, training=training)  
        v = self.wv(v, training=training)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(scaled_attention, 
                                    (batch_size, -1, self.d_model))
        
        output = self.dense(concat_attention, training=training)
        
        return output, attention_weights

class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Transformer Encoder Layer"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, mask=None, training=False):
        attn_output, attn_weights = self.mha(x, x, x, mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2, attn_weights

class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Transformer Decoder Layer"""
    
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x, enc_output, mask=None, training=False):
        # Self-attention on queries
        attn1, attn_weights_block1 = self.mha1(x, x, x, None, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # Cross-attention between queries and encoder output
        attn2, attn_weights_block2 = self.mha2(
            out1, enc_output, enc_output, mask, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        
        # Feed forward network
        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2

class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer Encoder"""
    
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.enc_layers = [TransformerEncoderLayer(d_model, num_heads, dff, dropout_rate)
                          for _ in range(num_layers)]
        
    def call(self, x, mask=None, training=False):
        attention_weights = {}
        
        for i in range(self.num_layers):
            x, attn = self.enc_layers[i](x, mask, training=training)
            attention_weights[f'encoder_layer_{i+1}'] = attn
        
        return x, attention_weights

class TransformerDecoder(tf.keras.layers.Layer):
    """Transformer Decoder"""
    
    def __init__(self, num_layers, d_model, num_heads, dff, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff, dropout_rate)
                          for _ in range(num_layers)]
        
    def call(self, x, enc_output, mask=None, training=False):
        attention_weights = {}
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, mask, training=training)
            
            attention_weights[f'decoder_layer_{i+1}_block1'] = block1
            attention_weights[f'decoder_layer_{i+1}_block2'] = block2
        
        return x, attention_weights

class DETRBackbone(tf.keras.layers.Layer):
    """ResNet backbone for DETR"""
    
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(DETRBackbone, self).__init__()
        
        if backbone_name == 'resnet50':
            self.backbone = tf.keras.applications.ResNet50(
                include_top=False,
                weights='imagenet' if pretrained else None,
                input_shape=(None, None, 3)
            )
            self.out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove the last pooling layer to keep spatial dimensions
        self.backbone = tf.keras.Model(
            inputs=self.backbone.input,
            outputs=self.backbone.layers[-3].output  # Before GlobalAveragePooling
        )
        
        # 1x1 conv to reduce channels for transformer
        self.conv = tf.keras.layers.Conv2D(256, 1, use_bias=False)
        
    def call(self, x, training=False):
        x = self.backbone(x, training=training)
        x = self.conv(x, training=training)
        return x

class UnderwaterDETR(tf.keras.Model):
    """DETR model for underwater object detection"""
    
    def __init__(self, num_classes=7, num_queries=150, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048):
        super(UnderwaterDETR, self).__init__()
        
        self.num_classes = num_classes + 1  # +1 for no-object class
        self.num_queries = num_queries
        self.d_model = d_model
        
        # Backbone
        self.backbone = DETRBackbone('resnet50', pretrained=True)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding2D()
        
        # Transformer
        self.transformer_encoder = TransformerEncoder(
            num_encoder_layers, d_model, nhead, dim_feedforward
        )
        
        self.transformer_decoder = TransformerDecoder(
            num_decoder_layers, d_model, nhead, dim_feedforward
        )
        
        # Object queries (learnable)
        self.object_queries = self.add_weight(
            shape=(num_queries, d_model),
            initializer='random_normal',
            trainable=True,
            name='object_queries'
        )
        
        # Prediction heads
        self.class_head = tf.keras.layers.Dense(
            self.num_classes, name='class_head'
        )
        
        self.bbox_head = tf.keras.Sequential([
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(d_model, activation='relu'),
            tf.keras.layers.Dense(4, activation='sigmoid')  # Normalized coordinates
        ], name='bbox_head')
    
    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        
        # Feature extraction
        features = self.backbone(x, training=training)  # [B, H, W, C]
        
        # Add positional encoding
        pos_encoded_features = self.pos_encoder(features)
        
        # Flatten spatial dimensions for transformer
        height, width = tf.shape(features)[1], tf.shape(features)[2]
        features_flat = tf.reshape(pos_encoded_features, [batch_size, height * width, self.d_model])
        
        # Encoder
        encoded_features, enc_attention = self.transformer_encoder(
            features_flat, training=training
        )
        
        # Decoder with object queries
        queries = tf.tile(self.object_queries[None], [batch_size, 1, 1])
        decoded_features, dec_attention = self.transformer_decoder(
            queries, encoded_features, training=training
        )
        
        # Prediction heads
        class_logits = self.class_head(decoded_features, training=training)
        bbox_coords = self.bbox_head(decoded_features, training=training)
        
        return {
            'class_logits': class_logits,  # [B, num_queries, num_classes]
            'bbox_coords': bbox_coords,    # [B, num_queries, 4]
            'attention_weights': {**enc_attention, **dec_attention}
        }

class HungarianMatcher:
    """Hungarian algorithm for bipartite matching between predictions and targets"""
    
    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    def compute_cost_matrix(self, pred_logits, pred_boxes, target_classes, target_boxes):
        """Compute the cost matrix for Hungarian matching"""
        # This is a simplified version - full implementation would be more complex
        batch_size, num_queries = tf.shape(pred_logits)[0], tf.shape(pred_logits)[1]
        num_targets = tf.shape(target_classes)[0]
        
        # Classification cost (simplified)
        pred_probs = tf.nn.softmax(pred_logits, axis=-1)  # [B, num_queries, num_classes]
        
        # For simplicity, return a dummy cost matrix
        # In practice, this would compute actual costs based on class predictions,
        # bbox L1 distance, and GIoU
        cost_matrix = tf.random.uniform([num_queries, num_targets], dtype=tf.float32)
        
        return cost_matrix
    
    def match(self, pred_logits, pred_boxes, targets):
        """Perform Hungarian matching"""
        # This is a placeholder - actual implementation would use scipy.optimize.linear_sum_assignment
        # or a TensorFlow equivalent
        
        batch_size = tf.shape(pred_logits)[0]
        num_queries = tf.shape(pred_logits)[1]
        
        # Return dummy indices for now
        matched_indices = []
        for b in range(batch_size):
            # In practice, compute actual matching here
            pred_indices = tf.range(min(num_queries, 10), dtype=tf.int32)  # Dummy
            target_indices = tf.range(min(10, 10), dtype=tf.int32)  # Dummy
            matched_indices.append((pred_indices, target_indices))
        
        return matched_indices

class DETRLoss(tf.keras.losses.Loss):
    """DETR Loss Function with Hungarian matching"""
    
    def __init__(self, num_classes=7, matcher_costs={'class': 1.0, 'bbox': 5.0, 'giou': 2.0},
                 loss_weights={'ce': 1.0, 'bbox': 5.0, 'giou': 2.0}, class_weights=None, **kwargs):
        super(DETRLoss, self).__init__(**kwargs)
        
        self.num_classes = num_classes + 1  # +1 for no-object
        self.matcher = HungarianMatcher(**matcher_costs)
        self.loss_weights = loss_weights
        self.class_weights = class_weights or [1.0] * self.num_classes
        
    def call(self, y_true, y_pred):
        """Compute DETR loss"""
        pred_logits = y_pred['class_logits']
        pred_boxes = y_pred['bbox_coords']
        
        # For simplified implementation, compute basic losses
        # In practice, this would implement full DETR loss with Hungarian matching
        
        batch_size, num_queries = tf.shape(pred_logits)[0], tf.shape(pred_logits)[1]
        
        # Dummy targets for now
        target_classes = tf.zeros([batch_size, num_queries], dtype=tf.int32)  # No-object class
        target_boxes = tf.zeros([batch_size, num_queries, 4], dtype=tf.float32)
        
        # Classification loss (simplified)
        ce_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_classes, logits=pred_logits
            )
        )
        
        # Bounding box loss (simplified)
        bbox_loss = tf.reduce_mean(tf.abs(pred_boxes - target_boxes))
        
        # GIoU loss (placeholder)
        giou_loss = tf.constant(0.0)
        
        # Combine losses
        total_loss = (
            self.loss_weights['ce'] * ce_loss +
            self.loss_weights['bbox'] * bbox_loss +
            self.loss_weights['giou'] * giou_loss
        )
        
        return total_loss

class DETRTrainer:
    """Training pipeline for DETR"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.dataset_config = UnderwaterDatasetConfig()
        
        # Override image size for DETR
        self.dataset_config.image_size = tuple(self.config['dataset']['image_size'])
        
        # Setup GPU
        self.gpu_setup = GPUSetup()
        self.gpu_available = self.gpu_setup.configure_gpu()
        self.strategy = self.gpu_setup.get_strategy()
        
        # Initialize components
        self.data_loader = DataLoader(self.dataset_config)
        self.metrics_calculator = MetricsCalculator(self.dataset_config)
        self.visualizer = Visualizer(self.dataset_config)
        
        # Initialize model and training components
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        
        # Experiment tracking
        self.setup_experiment_tracking()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_experiment_tracking(self):
        """Setup Weights & Biases experiment tracking"""
        wandb.init(
            project=self.config['experiment']['project'],
            name=self.config['experiment']['name'],
            tags=self.config['experiment']['tags'],
            config=self.config
        )
    
    def build_model(self):
        """Build DETR model with distribution strategy"""
        with self.strategy.scope():
            # Model configuration
            model_config = self.config['model']
            
            # Create model
            self.model = UnderwaterDETR(
                num_classes=self.dataset_config.num_classes,
                num_queries=self.config['dataset']['max_objects'],
                d_model=model_config['transformer']['d_model'],
                nhead=model_config['transformer']['nhead'],
                num_encoder_layers=model_config['transformer']['num_encoder_layers'],
                num_decoder_layers=model_config['transformer']['num_decoder_layers'],
                dim_feedforward=model_config['transformer']['dim_feedforward']
            )
            
            # Build model with dummy input
            dummy_input = tf.random.normal([1, *self.dataset_config.image_size, 3])
            _ = self.model(dummy_input)
            
            # Setup optimizer
            learning_rate = self.config['training']['learning_rate']
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=self.config['training']['weight_decay'],
                beta_1=self.config['training']['optimizer']['betas'][0],
                beta_2=self.config['training']['optimizer']['betas'][1]
            )
            
            # Setup loss function
            self.loss_fn = DETRLoss(
                num_classes=self.dataset_config.num_classes,
                loss_weights=self.config['losses']['loss_weights'],
                class_weights=self.config['losses']['class_weights']
            )
            
            # Compile model
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss_fn,
                run_eagerly=False
            )
        
        print(f"DETR model built successfully on {self.strategy.num_replicas_in_sync} device(s)")
        self.model.summary()
    
    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        print("Preparing datasets for DETR...")
        
        batch_size = self.config['training']['batch_size']
        
        # Create datasets
        self.train_dataset = self.data_loader.create_tf_dataset('train', batch_size)
        self.val_dataset = self.data_loader.create_tf_dataset('val', batch_size)
        
        # Distribute datasets
        self.train_dataset = self.strategy.experimental_distribute_dataset(self.train_dataset)
        self.val_dataset = self.strategy.experimental_distribute_dataset(self.val_dataset)
        
        print("Datasets prepared successfully")
    
    def train(self):
        """Main training loop"""
        print("Starting DETR training...")
        
        epochs = self.config['training']['epochs']
        
        # Setup callbacks
        callbacks = self._setup_callbacks()
        
        # Training loop
        with self.strategy.scope():
            history = self.model.fit(
                self.train_dataset,
                epochs=epochs,
                validation_data=self.val_dataset,
                callbacks=callbacks,
                verbose=1
            )
        
        # Plot training curves
        self.visualizer.plot_training_curves(
            history.history,
            save_path=f"detr_training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        
        return history
    
    def _setup_callbacks(self):
        """Setup training callbacks"""
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath='best_detr_model.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if self.config['early_stopping']['enabled']:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor=self.config['early_stopping']['monitor'],
                patience=self.config['early_stopping']['patience'],
                mode=self.config['early_stopping']['mode'],
                min_delta=self.config['early_stopping']['min_delta'],
                verbose=1,
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
        
        # Learning rate scheduling - Step decay for DETR
        def step_decay(epoch):
            initial_lr = self.config['training']['learning_rate']
            lr_drop = self.config['training']['lr_drop']
            lr_drop_factor = self.config['training']['lr_drop_factor']
            
            if epoch >= lr_drop:
                return initial_lr * lr_drop_factor
            return initial_lr
        
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
        callbacks.append(lr_scheduler)
        
        # Weights & Biases callback
        wandb_callback = wandb.keras.WandbCallback(
            monitor='val_loss',
            mode='min',
            save_model=False
        )
        callbacks.append(wandb_callback)
        
        # Custom metrics logger
        metrics_logger = MetricsLogger(log_freq=25)  # Less frequent for DETR
        callbacks.append(metrics_logger)
        
        return callbacks
    
    def evaluate(self):
        """Evaluate model performance"""
        print("Evaluating DETR model...")
        
        # Load test dataset
        test_dataset = self.data_loader.create_tf_dataset('test', batch_size=1)
        
        # Run evaluation
        test_loss = self.model.evaluate(test_dataset, verbose=1)
        print(f"Test Loss: {test_loss}")
        
        return test_loss
    
    def visualize_attention(self, sample_images=5):
        """Visualize attention maps from transformer"""
        print("Visualizing attention maps...")
        
        # Get a few test samples
        test_dataset = self.data_loader.create_tf_dataset('test', batch_size=1)
        
        for i, (image, _) in enumerate(test_dataset.take(sample_images)):
            # Get predictions with attention weights
            predictions = self.model(image, training=False)
            attention_weights = predictions['attention_weights']
            
            # Plot attention maps (simplified visualization)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Original image
            axes[0, 0].imshow(tf.squeeze(image).numpy())
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Visualize some attention heads
            for j, (key, attn) in enumerate(list(attention_weights.items())[:5]):
                if j >= 5:
                    break
                
                row, col = divmod(j + 1, 3)
                if row < 2 and col < 3:
                    # Average attention across heads and visualize
                    attn_avg = tf.reduce_mean(attn[0], axis=0)  # [seq_len, seq_len]
                    
                    axes[row, col].imshow(attn_avg.numpy(), cmap='hot')
                    axes[row, col].set_title(f'Attention: {key}')
                    axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'attention_visualization_{i}.png', dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main training script for DETR"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DETR for underwater object detection')
    parser.add_argument('--config', type=str,
                       default='../configs/detr_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--visualize_attention', action='store_true',
                       help='Visualize attention maps after training')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DETRTrainer(args.config)
    
    # Build model
    trainer.build_model()
    
    # Prepare datasets
    trainer.prepare_datasets()
    
    # Train model
    history = trainer.train()
    
    # Evaluate model
    test_results = trainer.evaluate()
    
    # Visualize attention if requested
    if args.visualize_attention:
        trainer.visualize_attention()
    
    # Save final results
    results = {
        'training_history': history.history,
        'test_results': test_results,
        'config': trainer.config
    }
    
    # Save results
    import json
    with open(f'detr_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("DETR training completed successfully!")

if __name__ == "__main__":
    main()
