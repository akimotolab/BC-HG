import importlib.metadata
import subprocess

import flax
import jax
import jax.numpy as jnp
import jaxopt
import optax
import orbax.checkpoint

def get_version(module, module_name):
    """Get the version of a module."""
    try:
        return getattr(module, '__version__')
    except AttributeError:
        try:
            return importlib.metadata.version(module_name.lower())
        except Exception:
            return "Unknown"


def print_device_info():
    print("=== JAX Device Info ===")
    devices = jax.devices()
    print("Available devices:", devices)
    print("Default backend:", jax.default_backend())
    print("Available platforms:", [device.platform for device in devices])

    try:
        gpu_devices = jax.devices("gpu")
        print("GPU available:", len(gpu_devices) > 0)
        print("GPU devices:", gpu_devices)
    except RuntimeError as error:
        print("GPU not available:", str(error))

    if len(devices) > 0:
        x = jnp.array([1.0, 2.0, 3.0])
        print(f"Test array device: {x.device()}")
        result = jnp.sum(x * x)
        print(f"Test calculation result: {result}")
        print(f"Result device: {result.device()}")

def main():
    print("=== Package Versions ===")
    print(f"JAX: {jax.__version__}")
    print(f"JAXopt: {get_version(jaxopt, 'jaxopt')}")
    print(f"Orbax: {orbax.checkpoint.__version__}")
    print(f"Optax: {optax.__version__}")
    print(f"Flax: {flax.__version__}")

    print()
    print_device_info()

    print("\n=== Compatibility Check ===")
    try:
        x = jnp.array([1, 2, 3])
        print("✅ JAX basic operations work")

        solver = jaxopt.GradientDescent(lambda value: jnp.sum(value**2))
        print("✅ JAXopt imports successfully")

        from orbax.checkpoint import PyTreeCheckpointer
        _ = PyTreeCheckpointer
        _ = solver
        _ = x
        print("✅ Orbax checkpoint imports successfully")
    except Exception as error:
        print(f"❌ Error: {error}")

    print("\n=== Pip Package Versions ===")
    try:
        result = subprocess.run(["pip", "show", "jaxopt"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    print(f"JAXopt (pip): {line.split(':')[1].strip()}")
                    break
        else:
            print("JAXopt: Not found via pip")
    except Exception as error:
        print(f"Error checking pip: {error}")


if __name__ == "__main__":
    main()