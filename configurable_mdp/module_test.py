# version_check.py
import jax
import jaxopt
import orbax.checkpoint
import optax
import flax

def get_version(module, module_name):
    """モジュールのバージョンを取得する関数"""
    try:
        return getattr(module, '__version__')
    except AttributeError:
        try:
            # 別の方法でバージョンを取得
            import importlib.metadata
            return importlib.metadata.version(module_name.lower())
        except Exception:
            return "Unknown"

print("=== Package Versions ===")
print(f"JAX: {jax.__version__}")
print(f"JAXopt: {get_version(jaxopt, 'jaxopt')}")
print(f"Orbax: {orbax.checkpoint.__version__}")
print(f"Optax: {optax.__version__}")
print(f"Flax: {flax.__version__}")

print("\n=== Compatibility Check ===")
try:
    # JAX API確認
    import jax.numpy as jnp
    x = jnp.array([1, 2, 3])
    print("✅ JAX basic operations work")
    
    # JAXopt確認
    import jaxopt
    solver = jaxopt.GradientDescent(lambda x: jnp.sum(x**2))
    print("✅ JAXopt imports successfully")
    
    # Orbax確認  
    from orbax.checkpoint import PyTreeCheckpointer
    print("✅ Orbax checkpoint imports successfully")
    
except Exception as e:
    print(f"❌ Error: {e}")

# 追加：pipでインストールされたバージョンを確認
print("\n=== Pip Package Versions ===")
import subprocess
try:
    result = subprocess.run(['pip', 'show', 'jaxopt'], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                print(f"JAXopt (pip): {line.split(':')[1].strip()}")
                break
    else:
        print("JAXopt: Not found via pip")
except Exception as e:
    print(f"Error checking pip: {e}")