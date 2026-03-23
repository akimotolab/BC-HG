import importlib
import importlib.metadata
import subprocess
import sys

def get_version(module, package_name):
    try:
        return getattr(module, "__version__")
    except AttributeError:
        try:
            return importlib.metadata.version(package_name)
        except Exception:
            return "Unknown"


def check_import(module_name, label=None, package_name=None):
    display_name = label or module_name
    package = package_name or module_name
    try:
        module = importlib.import_module(module_name)
        version = get_version(module, package)
        print(f"✅ {display_name}: import OK (version={version})")
        return module, None
    except Exception as error:
        print(f"❌ {display_name}: import failed ({error})")
        return None, error


def check_core_packages():
    print("=== Core Package Import Check ===")
    targets = [
        ("jax", "JAX", "jax"),
        ("jaxlib", "jaxlib", "jaxlib"),
        ("jaxopt", "JAXopt", "jaxopt"),
        ("optax", "Optax", "optax"),
        ("flax", "Flax", "flax"),
        ("orbax.checkpoint", "Orbax Checkpoint", "orbax-checkpoint"),
        ("gymnax", "Gymnax", "gymnax"),
        ("yaml", "PyYAML", "PyYAML"),
        ("pandas", "Pandas", "pandas"),
        ("matplotlib", "Matplotlib", "matplotlib"),
    ]

    loaded = {}
    failures = []
    for module_name, label, package_name in targets:
        module, error = check_import(module_name, label, package_name)
        loaded[module_name] = module
        if error is not None:
            failures.append(label)

    return loaded, failures


def check_jax_runtime(loaded_modules):
    print("=== JAX Device Info ===")
    failures = []

    jax_module = loaded_modules.get("jax")
    if jax_module is None:
        print("❌ Skipped: JAX import failed")
        return ["JAX runtime"]

    try:
        jnp_module = importlib.import_module("jax.numpy")
    except Exception as error:
        print(f"❌ Failed to import jax.numpy: {error}")
        return ["jax.numpy"]

    try:
        devices = jax_module.devices()
        print("Available devices:", devices)
        print("Default backend:", jax_module.default_backend())
        print("Available platforms:", [device.platform for device in devices])
    except Exception as error:
        print(f"❌ Failed to query JAX devices: {error}")
        return ["JAX device query"]

    try:
        gpu_devices = jax_module.devices("gpu")
        print("GPU available:", len(gpu_devices) > 0)
        print("GPU devices:", gpu_devices)
    except RuntimeError as error:
        print("GPU not available:", str(error))

    try:
        input_vector = jnp_module.array([1.0, 2.0, 3.0])
        print(f"Test array device: {input_vector.device()}")
        squared_sum = jnp_module.sum(input_vector * input_vector)
        print(f"Test calculation result: {squared_sum}")
        print(f"Result device: {squared_sum.device()}")

        @jax_module.jit
        def quadratic_fn(values):
            return jnp_module.sum(values**2)

        gradient_fn = jax_module.grad(quadratic_fn)
        gradient_value = gradient_fn(jnp_module.array([1.0, -2.0, 3.0]))
        print(f"✅ JAX jit/grad works (grad={gradient_value})")
    except Exception as error:
        print(f"❌ JAX runtime operation failed: {error}")
        failures.append("JAX runtime operation")

    return failures


def check_orbax_api():
    print("\n=== Orbax API Check ===")
    failures = []
    try:
        from orbax.checkpoint import PyTreeCheckpointer

        _ = PyTreeCheckpointer
        print("✅ Orbax checkpoint API imports successfully")
    except Exception as error:
        print(f"❌ Orbax API check failed: {error}")
        failures.append("Orbax API")

    return failures


def check_local_modules():
    print("\n=== Local Project Module Check ===")
    targets = [
        "src.algorithms.value_iteration_and_prediction",
        "src.environments.ConfigurableFourRooms",
        "src.models.IncentiveModel",
        "train_four_rooms_bchg",
        "train_bilevel_lqr_bchg",
    ]
    failures = []
    for module_name in targets:
        _, error = check_import(module_name, module_name)
        if error is not None:
            failures.append(module_name)
    return failures


def check_pip_versions(package_names):
    print("\n=== Pip Package Version Check ===")
    for package_name in package_names:
        try:
            result = subprocess.run(["pip", "show", package_name], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"- {package_name}: not found via pip show")
                continue

            version = "Unknown"
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break
            print(f"- {package_name}: {version}")
        except Exception as error:
            print(f"- {package_name}: version check failed ({error})")

def main():
    print("=== Python Runtime ===")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    loaded_modules, import_failures = check_core_packages()

    failures = []
    failures.extend(import_failures)
    failures.extend(check_jax_runtime(loaded_modules))
    failures.extend(check_orbax_api())
    failures.extend(check_local_modules())

    check_pip_versions(["jax", "jaxlib", "flax", "optax", "jaxopt", "orbax-checkpoint", "gymnax"])

    print("\n=== Summary ===")
    if failures:
        unique_failures = sorted(set(failures))
        print("❌ Environment verification failed for:")
        for name in unique_failures:
            print(f" - {name}")
        print("\nPlease re-check `configurable_mdp/environment.yaml` and reinstall missing/broken packages.")
        raise SystemExit(1)

    print("✅ Environment verification passed. You can proceed to run configurable_mdp experiments.")


if __name__ == "__main__":
    main()