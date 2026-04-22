import logging
import os
import threading
from ultralytics import YOLO
import torch
import hashlib

logger = logging.getLogger(__name__)

MODEL_FOLDER = os.getenv('MODEL_FOLDER', os.path.join(os.path.dirname(__file__), '..', 'models'))
ACTIVE_MODEL_FILE = os.getenv('ACTIVE_MODEL_FILE', os.path.join(MODEL_FOLDER, 'active_model.txt'))

current_model = None
current_model_path = None
current_model_hash = None
current_device = 'cuda' if torch.cuda.is_available() else 'cpu'

_model_lock = threading.Lock()


def _compute_file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _load_model_unsafe(path=None):
    """Load model — pemanggil harus sudah memegang _model_lock."""
    global current_model, current_model_path, current_model_hash

    if path:
        current_model_path = path
    else:
        if os.path.exists(ACTIVE_MODEL_FILE):
            with open(ACTIVE_MODEL_FILE, 'r') as f:
                current_model_path = f.read().strip()
        else:
            current_model_path = os.path.join(MODEL_FOLDER, 'yolov8n.pt')

    if not os.path.exists(current_model_path):
        raise FileNotFoundError(f"Model path not found: {current_model_path}")

    logger.debug("Loading model from: %s", current_model_path)
    current_model = YOLO(current_model_path)
    current_model.to(current_device)
    current_model_hash = _compute_file_hash(current_model_path)
    logger.debug("Model loaded on device: %s", current_device)


def load_model(path=None):
    with _model_lock:
        _load_model_unsafe(path)


def get_model():
    global current_model, current_model_path, current_model_hash

    with _model_lock:
        active_model_path = None
        if os.path.exists(ACTIVE_MODEL_FILE):
            with open(ACTIVE_MODEL_FILE, 'r') as f:
                active_model_path = f.read().strip()
        else:
            active_model_path = os.path.join(MODEL_FOLDER, 'yolov8n.pt')

        try:
            new_hash = _compute_file_hash(active_model_path)
        except Exception as e:
            logger.error("Gagal menghitung hash model: %s", e)
            # Pertahankan hash lama agar tidak reload terus-menerus saat error
            new_hash = current_model_hash

        if (current_model is None or
                current_model_path != active_model_path or
                current_model_hash != new_hash):
            logger.info("Reloading model karena path atau isi model berubah.")
            _load_model_unsafe(active_model_path)

        return current_model


def clear_model_cache():
    global current_model, current_model_path, current_model_hash
    with _model_lock:
        current_model = None
        current_model_path = None
        current_model_hash = None
        logger.debug("Cache model dihapus")
