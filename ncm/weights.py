import os
import urllib.request
import sys

DEFAULT_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
RELEASE_BASE_URL = "https://github.com/corpcol/ncm-codec/releases/download/v0.1.0"

MODELS = {
    "4x": "codec_4x_best.pt",
    "2x": "codec_2x_best.pt"
}

def get_weights_path(model_type: str = "4x") -> str:
    """Returns local path to checkpoint, checking standard directories."""
    filename = MODELS.get(model_type, f"codec_{model_type}_best.pt")
    local_path = os.path.join(DEFAULT_WEIGHTS_DIR, filename)
    if os.path.exists(local_path):
        return os.path.abspath(local_path)
    
    # Check ~/.cache/ncm
    user_cache = os.path.join(os.path.expanduser("~"), ".cache", "ncm", filename)
    if os.path.exists(user_cache):
        return os.path.abspath(user_cache)
        
    return local_path

def download_weights(model_type: str = "4x", dest_dir: str = None) -> str:
    """Download pretrained weights from GitHub Releases."""
    filename = MODELS.get(model_type, f"codec_{model_type}_best.pt")
    dest_dir = dest_dir or os.path.join(os.path.expanduser("~"), ".cache", "ncm")
    os.makedirs(dest_dir, exist_ok=True)
    
    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        print(f"[*] Weights already present at {dest_path}")
        return dest_path
        
    url = f"{RELEASE_BASE_URL}/{filename}"
    print(f"[*] Downloading {model_type} weights from {url}...")
    
    def _progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r[{pct}%] {count*block_size / 2**20:.1f} MB / {total_size / 2**20:.1f} MB")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=_progress)
        print(f"\n[+] Successfully downloaded to {dest_path}")
        return dest_path
    except Exception as e:
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise RuntimeError(f"Failed to download weights from {url}: {e}")
