import os
import cv2
import hashlib
from flask import Blueprint, request, current_app, send_from_directory, jsonify, render_template
from werkzeug.utils import secure_filename
from menu_config import menus
from route.utils import mark_active_menu
from route.model_manager import get_model, load_model

segmentation_video_bp = Blueprint('segmentation_video', __name__, template_folder='../templates')

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def _compute_file_hash(path):
    """Hitung hash MD5 file untuk deteksi perubahan model"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


@segmentation_video_bp.route('/upload', methods=['GET', 'POST'])
def upload_video():
    menus_active = mark_active_menu(request.path, menus)

    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('segmentation_video_upload.html', menus=menus_active, title="Upload Video - Segmentation", error='Tidak ada file video diupload')

        file = request.files['video']
        if file.filename == '':
            return render_template('segmentation_video_upload.html', menus=menus_active, title="Upload Video - Segmentation", error='Nama file kosong')

        if not allowed_file(file.filename):
            return render_template('segmentation_video_upload.html', menus=menus_active, title="Upload Video - Segmentation", error='Format file tidak didukung')

        filename = secure_filename(file.filename)
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        result_folder = current_app.config.get('RESULT_FOLDER', 'static/results')

        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(result_folder, exist_ok=True)

        upload_path = os.path.join(upload_folder, filename)
        file.save(upload_path)

        # Cek apakah model sudah berubah (otomatis reload)
        active_model_path = None
        model_hash_before = None
        model_hash_after = None
        active_model_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'active_model.txt')
        if os.path.exists(active_model_file):
            with open(active_model_file, 'r') as f:
                active_model_path = f.read().strip()
            if os.path.exists(active_model_path):
                model_hash_before = _compute_file_hash(active_model_path)

        # Pastikan model aktif sudah diload / reload kalau berubah
        try:
            load_model()
        except Exception as e:
            return render_template('segmentation_video_upload.html', menus=menus_active, title="Upload Video - Segmentation", error=f'Gagal load model: {e}')

        # Hitung ulang hash model setelah load_model
        if active_model_path and os.path.exists(active_model_path):
            model_hash_after = _compute_file_hash(active_model_path)
            if model_hash_before != model_hash_after:
                print("[INFO] Model berubah, sudah direload otomatis.")

        # Proses inference video untuk segmentation
        result_filename = f'segmented_{filename}'
        result_path = os.path.join(result_folder, result_filename)

        model = get_model()
        cap = cv2.VideoCapture(upload_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)
            frame_with_mask = results[0].plot()
            out.write(frame_with_mask)

        cap.release()
        out.release()

        return render_template('segmentation_video_upload.html',
                               menus=menus_active,
                               title="Upload Video - Segmentation",
                               message="Video berhasil diproses.",
                               result_video_url=f'/segmentation/video/download/{result_filename}',
                               result_video_name=result_filename)

    # GET method - tampilkan form upload
    return render_template('segmentation_video_upload.html', menus=menus_active, title="Upload Video - Segmentation")


@segmentation_video_bp.route('/download/<filename>')
def download_result(filename):
    result_folder = current_app.config.get('RESULT_FOLDER', 'static/results')
    return send_from_directory(result_folder, filename, as_attachment=True)
