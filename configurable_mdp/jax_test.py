import jax
import jax.numpy as jnp

# 利用可能なデバイスを表示
print("Available devices:", jax.devices())

# デフォルトのバックエンドを表示
print("Default backend:", jax.default_backend())

# 利用可能なプラットフォームを表示
print("Available platforms:", [device.platform for device in jax.devices()])

# GPUが利用可能かチェック（エラーを回避）
try:
    gpu_devices = jax.devices('gpu')
    print("GPU available:", len(gpu_devices) > 0)
    print("GPU devices:", gpu_devices)
except RuntimeError as e:
    print("GPU not available:", str(e))

# JAXのバージョン確認
print("JAX version:", jax.__version__)

# 簡単な計算テスト
if len(jax.devices()) > 0:
    x = jnp.array([1.0, 2.0, 3.0])
    print(f"Test array device: {x.device()}")
    result = jnp.sum(x * x)
    print(f"Test calculation result: {result}")
    print(f"Result device: {result.device()}")