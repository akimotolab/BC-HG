import importlib
import importlib.metadata
from pathlib import Path
import subprocess
import sys


def get_version_from_module(module, package_name):
    try:
        return getattr(module, "__version__")
    except AttributeError:
        try:
            return importlib.metadata.version(package_name)
        except Exception:
            return "Unknown"


def check_import(module_name, label=None):
    name = label or module_name
    try:
        module = importlib.import_module(module_name)
        version = get_version_from_module(module, module_name)
        print(f"✅ {name}: import OK (version={version})")
        return module, None
    except Exception as error:
        print(f"❌ {name}: import failed ({error})")
        return None, error


def check_python_version():
    print("=== Python Version Check ===")
    warnings = []
    major, minor = sys.version_info[:2]
    print(f"Detected Python: {major}.{minor}")
    if (major, minor) != (3, 8):
        warning = "Recommended Python is 3.8.x for `markov_game/environment.yaml` compatibility."
        print(f"⚠️ {warning}")
        warnings.append(warning)
    else:
        print("✅ Python version matches recommended 3.8.x")
    return warnings


def print_core_versions():
    print("=== Core Package Import Check ===")
    modules = {
        "numpy": "NumPy",
        "scipy": "SciPy",
        "pandas": "Pandas",
        "matplotlib": "Matplotlib",
        "omegaconf": "OmegaConf",
        "gym": "Gym",
        "tensorflow": "TensorFlow",
        "tensorboard": "TensorBoard",
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "torchaudio": "TorchAudio",
        "garage": "garage",
        "cloudpickle": "cloudpickle",
        "dill": "dill",
        "dowel": "dowel",
    }

    loaded = {}
    failures = []
    for module_name, label in modules.items():
        module, error = check_import(module_name, label)
        loaded[module_name] = module
        if error is not None:
            failures.append(label)

    return loaded, failures


def check_torch_runtime(torch_module):
    print("\n=== PyTorch Runtime Check ===")
    if torch_module is None:
        print("❌ Skipped: PyTorch import failed")
        return ["PyTorch runtime"]

    failures = []
    try:
        print(f"PyTorch version: {torch_module.__version__}")
        print(f"PyTorch CUDA runtime version: {torch_module.version.cuda}")
        cuda_available = torch_module.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        if cuda_available:
            device_count = torch_module.cuda.device_count()
            print(f"CUDA device count: {device_count}")
            for index in range(device_count):
                print(f" - GPU {index}: {torch_module.cuda.get_device_name(index)}")

            x = torch_module.randn(256, 256, device="cuda")
            y = torch_module.randn(256, 256, device="cuda")
            z = x @ y
            print(f"✅ CUDA tensor matmul OK (device={z.device}, mean={z.mean().item():.6f})")
        else:
            x = torch_module.randn(256, 256)
            y = torch_module.randn(256, 256)
            z = x @ y
            print(f"⚠️ CUDA unavailable; CPU tensor matmul OK (device={z.device}, mean={z.mean().item():.6f})")
    except Exception as error:
        print(f"❌ PyTorch runtime check failed: {error}")
        failures.append("PyTorch runtime")

    return failures


def check_garage_torch_api():
    print("\n=== garage + torch API Check ===")
    failures = []
    try:
        from garage.torch import prefer_gpu, set_gpu_mode
        from garage.torch.policies import DeterministicMLPPolicy, TanhGaussianMLPPolicy
        from garage.torch.q_functions import ContinuousMLPQFunction, DiscreteMLPQFunction

        _ = prefer_gpu
        _ = set_gpu_mode
        _ = DeterministicMLPPolicy
        _ = TanhGaussianMLPPolicy
        _ = ContinuousMLPQFunction
        _ = DiscreteMLPQFunction
        print("✅ garage.torch API imports successfully")
    except Exception as error:
        print(f"❌ garage.torch API check failed: {error}")
        failures.append("garage.torch API")

    return failures


def check_local_training_modules():
    print("\n=== Local Project Module Check ===")
    failures = []
    try:
        import src.experiment
        import src.envs
        import src.follower
        import src.policies
        import src.replay_buffer
        import src.sampler

        _ = src.experiment
        _ = src.envs
        _ = src.follower
        _ = src.policies
        _ = src.replay_buffer
        _ = src.sampler
        print("✅ local modules under markov_game/src import successfully")
    except Exception as error:
        print(f"❌ local module import failed: {error}")
        failures.append("local project modules")

    return failures


def check_training_script_imports():
    print("\n=== Training Script Import Check ===")
    failures = []
    targets = [
        "train_discrete_toy",
        "train_bilevel_lqr",
    ]
    for module_name in targets:
        _, error = check_import(module_name, module_name)
        if error is not None:
            failures.append(module_name)
    return failures


def check_required_files():
    print("\n=== Required File Check ===")
    failures = []
    required_paths = [
        Path("markov_game/environment.yaml"),
        Path("markov_game/config/config_discrete_toy_bchg.yaml"),
        Path("markov_game/config/config_bilevel_lqr_bchg.yaml"),
    ]
    for path in required_paths:
        if path.exists():
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
            failures.append(str(path))
    return failures


def check_nvidia_driver_visibility():
    print("\n=== NVIDIA Driver Visibility Check ===")
    warnings = []
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if lines:
                print("✅ nvidia-smi is available:")
                for line in lines:
                    print(f" - {line}")
            else:
                print("⚠️ nvidia-smi is available but no GPUs were listed")
        else:
            warning = "nvidia-smi returned non-zero; GPU driver/runtime may be unavailable in this shell."
            print(f"⚠️ {warning}")
            warnings.append(warning)
    except FileNotFoundError:
        warning = "nvidia-smi command not found; skip if running CPU-only or in restricted container."
        print(f"⚠️ {warning}")
        warnings.append(warning)
    return warnings


def check_pip_versions(package_names):
    print("\n=== Pip Package Version Check ===")
    for package_name in package_names:
        try:
            result = subprocess.run(
                ["pip", "show", package_name],
                capture_output=True,
                text=True,
                check=False,
            )
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

    warnings = []
    warnings.extend(check_python_version())

    loaded, import_failures = print_core_versions()

    failures = []
    failures.extend(import_failures)
    failures.extend(check_torch_runtime(loaded.get("torch")))
    failures.extend(check_garage_torch_api())
    failures.extend(check_local_training_modules())
    failures.extend(check_training_script_imports())
    failures.extend(check_required_files())

    warnings.extend(check_nvidia_driver_visibility())

    check_pip_versions(["garage", "torch", "tensorflow", "omegaconf"])

    print("\n=== Summary ===")
    if warnings:
        print("⚠️ Warnings:")
        for warning in sorted(set(warnings)):
            print(f" - {warning}")

    if failures:
        unique_failures = sorted(set(failures))
        print("❌ Environment verification failed for:")
        for name in unique_failures:
            print(f" - {name}")
        print("\nPlease re-check `markov_game/environment.yaml` and reinstall missing/broken packages.")
        raise SystemExit(1)

    print("✅ Environment verification passed. You can proceed to run markov_game experiments.")


if __name__ == "__main__":
    main()
