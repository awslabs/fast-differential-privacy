{
  "bf16": {
      "enabled": true
    },
  "fp16": {
      "enabled": false,
      "fp16_master_weights_and_grads": false,
      "loss_scale": 1,
      "loss_scale_window": 1000,
      "hysteresis": 2,
      "min_loss_scale": 1,
      "initial_scale_power": 3
  },
  "train_micro_batch_size_per_gpu": 999999999,
  "wall_clock_breakdown": false,
  "zero_optimization": {
      "stage": 1,
      "allgather_partitions": true,
      "reduce_scatter": true,
      "allgather_bucket_size": 50000000,
      "reduce_bucket_size": 50000000,
      "overlap_comm": true,
      "contiguous_gradients": true,
      "cpu_offload": false
  }
}
