import jax
import jax.numpy as jnp

# Show available devices
print("Available devices:", jax.devices())

# Show the default backend
print("Default backend:", jax.default_backend())

# Show available platforms
print("Available platforms:", [device.platform for device in jax.devices()])

# Check whether a GPU is available (avoid errors)
try:
    gpu_devices = jax.devices('gpu')
    print("GPU available:", len(gpu_devices) > 0)
    print("GPU devices:", gpu_devices)
except RuntimeError as e:
    print("GPU not available:", str(e))

# Check the JAX version
print("JAX version:", jax.__version__)

# Simple computation test
if len(jax.devices()) > 0:
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"Test array device: {x.device()}")
    result = jnp.sum(x * x)
    print(f"Test calculation result: {result}")
    print(f"Result device: {result.device()}")