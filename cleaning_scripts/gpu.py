import os
import tensorflow as tf


# Add this at the beginning of your training script
def setup_gpu():
    """Configure TensorFlow to use GPU and display GPU information."""
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')

    if not gpus:
        print("❌ No GPU found. Using CPU for training (will be slower).")
        return False

    # Print GPU information
    print(f"✅ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i + 1}: {gpu.name}")

    # Memory growth - prevents TensorFlow from allocating all GPU memory at once
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Set memory growth for {gpu.name}")
        except RuntimeError as e:
            print(f"❌ Error setting memory growth: {e}")

    # Optionally limit GPU memory usage if needed
    # tf.config.set_logical_device_configuration(
    #     gpus[0],
    #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Limit to 4GB
    # )

    # Verify GPU is being used
    print(f"✅ TensorFlow is using: {tf.config.get_visible_devices()}")
    print(f"✅ Built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"✅ GPU available: {tf.test.is_gpu_available()}")

    return True


# Call this function at the beginning of your main code
gpu_available = setup_gpu()