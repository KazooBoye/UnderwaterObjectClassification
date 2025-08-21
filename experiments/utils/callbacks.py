"""
Custom learning rate schedulers and callbacks for underwater object detection
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CosineRestartScheduler(tf.keras.callbacks.Callback):
    """Cosine Annealing with Warm Restarts (SGDR)"""
    
    def __init__(self, first_restart_step, t_mul=2.0, m_mul=0.8, alpha=0.01):
        super(CosineRestartScheduler, self).__init__()
        self.first_restart_step = first_restart_step
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = alpha
        
        self.step = 0
        self.t_cur = 0
        self.t_i = first_restart_step
        self.eta_max = None
        self.eta_min = None
        
    def on_train_begin(self, logs=None):
        self.eta_max = self.model.optimizer.learning_rate.numpy()
        self.eta_min = self.eta_max * self.alpha
        
    def on_epoch_begin(self, epoch, logs=None):
        self.step = epoch
        
        # Check if we need to restart
        if self.t_cur >= self.t_i:
            self.t_cur = 0
            self.t_i = int(self.t_i * self.t_mul)
            self.eta_max *= self.m_mul
            self.eta_min = self.eta_max * self.alpha
        
        # Calculate current learning rate
        eta_t = self.eta_min + (self.eta_max - self.eta_min) * \
                (1 + np.cos(np.pi * self.t_cur / self.t_i)) / 2
        
        # Update learning rate
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, eta_t)
        
        self.t_cur += 1
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = self.model.optimizer.learning_rate.numpy()

class WarmupCosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with warmup and cosine decay"""
    
    def __init__(self, warmup_steps, total_steps, peak_lr, min_lr=1e-6):
        super(WarmupCosineSchedule, self).__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        
        # Warmup phase
        warmup_lr = self.peak_lr * step / warmup_steps
        
        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / decay_steps))
        decay_lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        
        return tf.where(step <= warmup_steps, warmup_lr, decay_lr)

class MetricsLogger(tf.keras.callbacks.Callback):
    """Custom metrics logger for object detection"""
    
    def __init__(self, log_freq=10):
        super(MetricsLogger, self).__init__()
        self.log_freq = log_freq
        self.metrics_history = {
            'loss': [],
            'val_loss': [],
            'lr': []
        }
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Store metrics
        for key, value in logs.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)
        
        # Log learning rate
        lr = self.model.optimizer.learning_rate.numpy()
        self.metrics_history['lr'].append(lr)
        
        # Print detailed metrics
        if (epoch + 1) % self.log_freq == 0:
            print(f"\nEpoch {epoch + 1} Metrics:")
            print(f"  Learning Rate: {lr:.2e}")
            for key, value in logs.items():
                if 'val_' in key:
                    print(f"  {key}: {value:.4f}")
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        if 'loss' in self.metrics_history and 'val_loss' in self.metrics_history:
            axes[0, 0].plot(self.metrics_history['loss'], label='Training')
            axes[0, 0].plot(self.metrics_history['val_loss'], label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
        
        # Learning rate
        if 'lr' in self.metrics_history:
            axes[0, 1].plot(self.metrics_history['lr'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class EMACallback(tf.keras.callbacks.Callback):
    """Exponential Moving Average callback for model weights"""
    
    def __init__(self, decay=0.9999, start_epoch=0):
        super(EMACallback, self).__init__()
        self.decay = decay
        self.start_epoch = start_epoch
        self.ema_weights = None
        
    def on_train_begin(self, logs=None):
        # Initialize EMA weights
        self.ema_weights = [tf.Variable(w, trainable=False) for w in self.model.weights]
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.start_epoch:
            # Update EMA weights
            for ema_w, w in zip(self.ema_weights, self.model.weights):
                ema_w.assign(self.decay * ema_w + (1 - self.decay) * w)
    
    def apply_ema_weights(self):
        """Apply EMA weights to model"""
        for ema_w, w in zip(self.ema_weights, self.model.weights):
            w.assign(ema_w)
    
    def restore_original_weights(self, original_weights):
        """Restore original weights"""
        for w, orig_w in zip(self.model.weights, original_weights):
            w.assign(orig_w)

class GradientLoggingCallback(tf.keras.callbacks.Callback):
    """Log gradient statistics during training"""
    
    def __init__(self, log_freq=10):
        super(GradientLoggingCallback, self).__init__()
        self.log_freq = log_freq
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_freq == 0:
            # Get gradients (this is a simplified version)
            with tf.GradientTape() as tape:
                # This would need to be integrated with actual training step
                pass
