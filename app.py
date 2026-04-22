import sys
from dotenv import load_dotenv
import os
import shutil
import threading
import time

# Load variabel lingkungan dari .env sebelum menginisialisasi aplikasi
load_dotenv()

# Tambahkan path direktori saat ini ke sys.path untuk mengimpor modul internal
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask
from menu_config import menus
from route.main_routes import main_bp
from route.setting_routes import setting_bp
from route.object_detection_routes import obj_detect_bp
from route.object_detection_file_multi_routes import obj_detect_file_multi_bp
from route.video_detection_routes import video_detect_bp
from route.segmentation_routes import segmentation_bp
from route.segmentation_file_multi_routes import segmentation_file_multi_bp
from route.segmentation_video_routes import segmentation_video_bp
from route.experimental_routes import experimental_bp
from route.data_routes import data_bp
from route.experimental_auto_annotation_sahi_routes import experimental_sahi_bp
from route.object_tracker_routes import object_tracker_bp
from route.object_tracker_segmentation_routes import bp as object_tracker_segmentation_bp

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Ambil konfigurasi dari variabel lingkungan (.env)
app.secret_key = os.getenv('SECRET_KEY', 'default-secret-key')

# Konfigurasi folder dari variabel lingkungan
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'static/uploads')
app.config['RESULT_FOLDER'] = os.getenv('RESULT_FOLDER', 'static/results')
app.config['ANNOTATION_FOLDER'] = os.getenv('ANNOTATION_FOLDER', 'static/annotations')
app.config['DATASET_FOLDER'] = os.getenv('DATASET_FOLDER', 'static/dataset')

# Pastikan folder ada, buat jika belum
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['ANNOTATION_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)

def clear_folder(folder_path):
    """Hapus semua isi folder, tapi tidak menghapus folder itu sendiri"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # hapus file / symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # hapus folder dan isinya
        except Exception as e:
            print(f'Gagal menghapus {file_path}. Alasan: {e}')

def scheduled_clear_folders(interval_seconds=3600):
    """Thread background untuk hapus isi folder setiap interval detik (default 1 jam)"""
    while True:
        time.sleep(interval_seconds)
        print("[INFO] Scheduled cleaning: menghapus isi folder upload, result, annotations, dan dataset...")
        clear_folder(app.config['UPLOAD_FOLDER'])
        clear_folder(app.config['RESULT_FOLDER'])
        clear_folder(app.config['ANNOTATION_FOLDER'])
        clear_folder(app.config['DATASET_FOLDER'])

# Bersihkan folder-folder penting saat server mulai
clear_folder(app.config['UPLOAD_FOLDER'])
clear_folder(app.config['RESULT_FOLDER'])
clear_folder(app.config['ANNOTATION_FOLDER'])
clear_folder(app.config['DATASET_FOLDER'])

# Jalankan thread background hapus folder setiap 1 jam (3600 detik)
threading.Thread(target=scheduled_clear_folders, args=(3600,), daemon=True).start()

# Daftarkan blueprint
app.register_blueprint(main_bp)
app.register_blueprint(setting_bp)
app.register_blueprint(obj_detect_bp, url_prefix='/object-detection')
app.register_blueprint(obj_detect_file_multi_bp, url_prefix='/object-detection/file-multi')
app.register_blueprint(video_detect_bp, url_prefix='/object-detection/video')
app.register_blueprint(segmentation_bp, url_prefix='/segmentation')
app.register_blueprint(segmentation_file_multi_bp, url_prefix='/segmentation/image')
app.register_blueprint(segmentation_video_bp, url_prefix='/segmentation/video')
app.register_blueprint(experimental_bp)
app.register_blueprint(data_bp)
app.register_blueprint(experimental_sahi_bp)
app.register_blueprint(object_tracker_bp, url_prefix='/object-tracker')
app.register_blueprint(object_tracker_segmentation_bp)

if __name__ == '__main__':
    # Mode debug diambil dari variabel lingkungan
    debug_mode = os.getenv('DEBUG', 'False') == 'True'
    app.run(debug=debug_mode)