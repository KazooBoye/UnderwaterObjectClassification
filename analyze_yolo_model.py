#!/usr/bin/env python3
"""
YOLO Model Analysis - Parameters, Size, and Compute Requirements
Analyzes the YOLOv8-style TensorFlow model for resource estimation
"""

import numpy as np
from typing import Dict, Tuple

class YOLOModelAnalyzer:
    """Analyze YOLO model complexity and resource requirements"""
    
    def __init__(self):
        self.input_size = 640
        self.num_classes = 7
        self.batch_size = 16
        
    def analyze_backbone_parameters(self) -> Dict[str, int]:
        """Calculate CSPDarknet backbone parameters"""
        params = {}
        
        # Stem layers
        # Conv2D(32, 3x3) + BN + Activation
        params['stem_conv1'] = (3 * 32 * 3 * 3) + 32  # weights + bias
        params['stem_bn1'] = 32 * 4  # gamma, beta, moving_mean, moving_var
        
        # Conv2D(64, 3x3) + BN + Activation  
        params['stem_conv2'] = (32 * 64 * 3 * 3) + 64
        params['stem_bn2'] = 64 * 4
        
        # CSP Stages
        # Stage 1: 64->128, 3 blocks
        stage1_params = self._calculate_csp_stage_params(64, 128, 3)
        params['stage1'] = stage1_params
        
        # Stage 2: 128->256, 6 blocks
        stage2_params = self._calculate_csp_stage_params(128, 256, 6)
        params['stage2'] = stage2_params
        
        # Stage 3: 256->512, 9 blocks  
        stage3_params = self._calculate_csp_stage_params(256, 512, 9)
        params['stage3'] = stage3_params
        
        # Stage 4: 512->1024, 3 blocks
        stage4_params = self._calculate_csp_stage_params(512, 1024, 3)
        params['stage4'] = stage4_params
        
        return params
    
    def _calculate_csp_stage_params(self, in_ch: int, out_ch: int, num_blocks: int) -> int:
        """Calculate parameters for one CSP stage"""
        # Initial conv
        initial_conv = (in_ch * out_ch * 3 * 3) + (out_ch * 4)  # conv + bn
        
        # CSP blocks
        block_params = 0
        for _ in range(num_blocks):
            # Conv 1x1 (out_ch -> out_ch//2)
            conv1 = (out_ch * (out_ch//2) * 1 * 1) + ((out_ch//2) * 4)
            # Conv 3x3 (out_ch//2 -> out_ch//2)
            conv2 = ((out_ch//2) * (out_ch//2) * 3 * 3) + ((out_ch//2) * 4)
            # Conv 1x1 (out_ch//2 -> out_ch)
            conv3 = ((out_ch//2) * out_ch * 1 * 1) + (out_ch * 4)
            
            block_params += conv1 + conv2 + conv3
        
        return initial_conv + block_params
    
    def analyze_fpn_parameters(self) -> Dict[str, int]:
        """Calculate FPN parameters"""
        params = {}
        
        # Lateral convolutions
        params['lateral_conv1'] = (512 * 512 * 1 * 1)  # c4->512
        params['lateral_conv2'] = (256 * 256 * 1 * 1)  # c3->256
        params['reduce_conv'] = (1024 * 512 * 1 * 1)   # c5->512
        
        # Bottom-up convolutions
        params['downsample_conv1'] = (256 * 512 * 3 * 3) + (512 * 4)
        params['downsample_conv2'] = (512 * 512 * 3 * 3) + (512 * 4)
        
        # Output convolutions
        # P3: 256 channels
        params['output_conv1'] = (256 * 256 * 3 * 3) + (256 * 4)
        # P4: 512 channels  
        params['output_conv2'] = (512 * 512 * 3 * 3) + (512 * 4)
        # P5: 1024 channels
        params['output_conv3'] = (1024 * 1024 * 3 * 3) + (1024 * 4)
        
        return params
    
    def analyze_head_parameters(self) -> Dict[str, int]:
        """Calculate detection head parameters"""
        params = {}
        
        # Each head processes 3 scales
        for scale_idx, channels in enumerate([256, 512, 1024]):
            # Shared convolutions
            shared_conv1 = (channels * 256 * 3 * 3) + (256 * 4)
            shared_conv2 = (256 * 256 * 3 * 3) + (256 * 4)
            
            # Output: 3 anchors * (5 + num_classes)
            output_channels = 3 * (5 + self.num_classes)  # 3 * 12 = 36
            output_conv = (256 * output_channels * 1 * 1) + output_channels
            
            params[f'head_scale_{scale_idx}'] = shared_conv1 + shared_conv2 + output_conv
        
        return params
    
    def calculate_total_parameters(self) -> Dict[str, int]:
        """Calculate total model parameters"""
        backbone_params = self.analyze_backbone_parameters()
        fpn_params = self.analyze_fpn_parameters()
        head_params = self.analyze_head_parameters()
        
        # Sum all parameters
        total_backbone = sum(backbone_params.values())
        total_fpn = sum(fpn_params.values())
        total_head = sum(head_params.values())
        total_params = total_backbone + total_fpn + total_head
        
        return {
            'backbone': total_backbone,
            'fpn': total_fpn,
            'head': total_head,
            'total': total_params
        }
    
    def calculate_model_size(self, total_params: int) -> Dict[str, float]:
        """Calculate model size in different formats"""
        # Assuming float32 (4 bytes per parameter)
        size_bytes = total_params * 4
        
        return {
            'bytes': size_bytes,
            'kb': size_bytes / 1024,
            'mb': size_bytes / (1024 * 1024),
            'gb': size_bytes / (1024 * 1024 * 1024)
        }
    
    def calculate_memory_requirements(self) -> Dict[str, float]:
        """Calculate memory requirements during training"""
        params = self.calculate_total_parameters()['total']
        
        # Memory components (in MB)
        memory_req = {}
        
        # Model parameters (float32)
        memory_req['model_params'] = (params * 4) / (1024 * 1024)
        
        # Gradients (same size as parameters)  
        memory_req['gradients'] = memory_req['model_params']
        
        # Optimizer state (Adam: 2x params for momentum and velocity)
        memory_req['optimizer_state'] = memory_req['model_params'] * 2
        
        # Forward pass activations (estimated)
        # Input: batch_size * 640 * 640 * 3 * 4 bytes
        input_size = self.batch_size * 640 * 640 * 3 * 4 / (1024 * 1024)
        
        # Feature maps at different scales (rough estimate)
        # P3: 80x80, P4: 40x40, P5: 20x20
        feature_maps = (
            self.batch_size * (80*80*256 + 40*40*512 + 20*20*1024) * 4 / (1024 * 1024)
        )
        
        memory_req['input_batch'] = input_size
        memory_req['feature_maps'] = feature_maps
        memory_req['misc_overhead'] = 200  # Buffer, framework overhead, etc.
        
        # Total training memory
        memory_req['total_training'] = sum(memory_req.values())
        
        # Inference memory (no gradients/optimizer)
        memory_req['inference_only'] = (
            memory_req['model_params'] + 
            memory_req['input_batch'] + 
            memory_req['feature_maps'] + 
            50  # Reduced overhead
        )
        
        return memory_req
    
    def calculate_compute_requirements(self) -> Dict[str, float]:
        """Calculate compute requirements (FLOPs and training time)"""
        compute = {}
        
        # FLOPs estimation for forward pass
        # Convolution FLOPs: output_h * output_w * kernel_h * kernel_w * in_channels * out_channels
        
        # Backbone FLOPs (rough estimation)
        backbone_flops = 0
        
        # Stem
        backbone_flops += 640 * 640 * 3 * 3 * 3 * 32  # First conv
        backbone_flops += 320 * 320 * 3 * 3 * 32 * 64  # Second conv
        
        # CSP stages (simplified)
        backbone_flops += 160 * 160 * 3 * 3 * 64 * 128 * 4  # Stage 1
        backbone_flops += 80 * 80 * 3 * 3 * 128 * 256 * 7   # Stage 2  
        backbone_flops += 40 * 40 * 3 * 3 * 256 * 512 * 10  # Stage 3
        backbone_flops += 20 * 20 * 3 * 3 * 512 * 1024 * 4  # Stage 4
        
        # FPN FLOPs
        fpn_flops = (
            20*20*1*1*1024*512 +  # Reduce conv
            40*40*1*1*512*512 +   # Lateral conv1
            80*80*1*1*256*256 +   # Lateral conv2
            40*40*3*3*256*512 +   # Downsample conv1  
            20*20*3*3*512*512 +   # Downsample conv2
            80*80*3*3*256*256 +   # Output conv1
            40*40*3*3*512*512 +   # Output conv2
            20*20*3*3*1024*1024   # Output conv3
        )
        
        # Head FLOPs (3 scales)
        head_flops = (
            80*80*3*3*256*256 + 80*80*3*3*256*256 + 80*80*1*1*256*36 +  # P3
            40*40*3*3*512*256 + 40*40*3*3*256*256 + 40*40*1*1*256*36 +  # P4
            20*20*3*3*1024*256 + 20*20*3*3*256*256 + 20*20*1*1*256*36   # P5
        )
        
        total_flops = backbone_flops + fpn_flops + head_flops
        
        compute['forward_pass_flops'] = total_flops
        compute['forward_pass_gflops'] = total_flops / 1e9
        
        # Training FLOPs (forward + backward ≈ 3x forward)
        compute['training_flops_per_batch'] = total_flops * 3
        compute['training_gflops_per_batch'] = compute['training_flops_per_batch'] / 1e9
        
        return compute
    
    def estimate_training_time(self, hardware_type: str = 'RTX3080') -> Dict[str, float]:
        """Estimate training time for different hardware"""
        
        hardware_specs = {
            'RTX4090': {'tflops': 165, 'memory_gb': 24},
            'RTX3080': {'tflops': 30, 'memory_gb': 10},
            'RTX2080': {'tflops': 14, 'memory_gb': 8},
            'V100': {'tflops': 112, 'memory_gb': 32},
            'CPU': {'tflops': 1, 'memory_gb': 64}
        }
        
        if hardware_type not in hardware_specs:
            hardware_type = 'RTX3080'
        
        specs = hardware_specs[hardware_type]
        compute = self.calculate_compute_requirements()
        
        # Training parameters
        epochs = 100
        samples_per_epoch = 448  # From your dataset
        batches_per_epoch = samples_per_epoch // self.batch_size
        total_batches = epochs * batches_per_epoch
        
        # Time estimation
        gflops_per_batch = compute['training_gflops_per_batch']
        theoretical_time_per_batch = gflops_per_batch / specs['tflops']  # seconds
        
        # Add overhead (data loading, loss computation, etc.)
        actual_time_per_batch = theoretical_time_per_batch * 2.5  # 150% overhead
        
        total_time_seconds = total_batches * actual_time_per_batch
        
        return {
            'hardware': hardware_type,
            'batches_per_epoch': batches_per_epoch,
            'total_batches': total_batches,
            'seconds_per_batch': actual_time_per_batch,
            'total_seconds': total_time_seconds,
            'total_minutes': total_time_seconds / 60,
            'total_hours': total_time_seconds / 3600,
            'gflops_per_batch': gflops_per_batch
        }

def main():
    """Run complete model analysis"""
    analyzer = YOLOModelAnalyzer()
    
    print("🔍 YOLO Model Analysis Report")
    print("=" * 60)
    
    # Parameter analysis
    params = analyzer.calculate_total_parameters()
    print(f"\n📊 Model Parameters:")
    print(f"  Backbone (CSPDarknet): {params['backbone']:,} parameters")
    print(f"  FPN (Feature Pyramid): {params['fpn']:,} parameters") 
    print(f"  Detection Heads:       {params['head']:,} parameters")
    print(f"  Total Parameters:      {params['total']:,} parameters")
    
    # Model size
    size = analyzer.calculate_model_size(params['total'])
    print(f"\n💾 Model Size:")
    print(f"  Float32 precision: {size['mb']:.1f} MB ({size['gb']:.3f} GB)")
    print(f"  Int8 quantized:    {size['mb']/4:.1f} MB (estimated)")
    
    # Memory requirements
    memory = analyzer.calculate_memory_requirements()
    print(f"\n🧠 Memory Requirements:")
    print(f"  Model parameters:  {memory['model_params']:.1f} MB")
    print(f"  Gradients:         {memory['gradients']:.1f} MB")
    print(f"  Optimizer state:   {memory['optimizer_state']:.1f} MB")
    print(f"  Input batch:       {memory['input_batch']:.1f} MB")
    print(f"  Feature maps:      {memory['feature_maps']:.1f} MB")
    print(f"  Overhead:          {memory['misc_overhead']:.1f} MB")
    print(f"  Total Training:    {memory['total_training']:.1f} MB ({memory['total_training']/1024:.1f} GB)")
    print(f"  Inference Only:    {memory['inference_only']:.1f} MB ({memory['inference_only']/1024:.1f} GB)")
    
    # Compute requirements
    compute = analyzer.calculate_compute_requirements()
    print(f"\n⚡ Compute Requirements:")
    print(f"  Forward pass:      {compute['forward_pass_gflops']:.2f} GFLOPs")
    print(f"  Training per batch: {compute['training_gflops_per_batch']:.2f} GFLOPs")
    
    # Training time estimates
    hardware_types = ['RTX4090', 'RTX3080', 'RTX2080', 'CPU']
    print(f"\n⏱️  Training Time Estimates (100 epochs):")
    
    for hw in hardware_types:
        timing = analyzer.estimate_training_time(hw)
        print(f"  {hw:8}: {timing['total_hours']:.1f} hours "
              f"({timing['seconds_per_batch']:.3f}s/batch, "
              f"{timing['total_batches']} batches)")
    
    # Hardware recommendations
    print(f"\n💡 Hardware Recommendations:")
    print(f"  Minimum GPU:    GTX 1660 (6GB) - Batch size 4-8")
    print(f"  Recommended:    RTX 3080 (10GB) - Batch size 16")  
    print(f"  Optimal:        RTX 4090 (24GB) - Batch size 32+")
    print(f"  Memory needed:  {memory['total_training']/1024:.1f} GB for training")
    
    # Optimization suggestions
    print(f"\n🚀 Optimization Tips:")
    print(f"  • Use mixed precision (FP16) to reduce memory by ~40%")
    print(f"  • Gradient accumulation for larger effective batch sizes")
    print(f"  • Model pruning can reduce parameters by 20-50%")
    print(f"  • Quantization (INT8) reduces size by 75% with minimal accuracy loss")

if __name__ == '__main__':
    main()
