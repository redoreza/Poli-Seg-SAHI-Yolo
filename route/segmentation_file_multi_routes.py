import os
import cv2
import zipfile
import hashlib
import io
from flask import Blueprint, render_template, request, current_app, jsonify, send_file
from werkzeug.utils import secure_filename
from menu_config import menus
from route.utils import mark_active_menu
from route.model_manager import get_model, load_model

segmentation_file_multi_bp = Blueprint(
    'segmentation_file_multi', __name__, template_folder='../templates'
)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def _compute_file_hash(path):
    """Hitung hash MD5 file untuk reload model jika berubah"""
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


@segmentation_file_multi_bp.route('/', methods=['GET'])
def segmentation_file_multi_index():
    menus_active = mark_active_menu(request.path, menus)
    return render_template(
        'segmentation_file_multi.html', menus=menus_active, title="Segmentation - Multiple File"
    )


@segmentation_file_multi_bp.route('/upload', methods=['POST'])
def segmentation_file_multi_upload():
    # Cek hash model sebelum reload
    active_model_path = None
    active_model_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'active_model.txt')
    if os.path.exists(active_model_file):
        with open(active_model_file, 'r') as f:
            active_model_path = f.read().strip()
    if active_model_path and os.path.exists(active_model_path):
        model_hash_before = _compute_file_hash(active_model_path)
    else:
        model_hash_before = None

    # Reload model otomatis jika berubah
    try:
        load_model()
    except Exception as e:
        return jsonify({'error': f'Gagal reload model: {e}'}), 500

    # Cek hash model setelah reload
    if active_model_path and os.path.exists(active_model_path):
        model_hash_after = _compute_file_hash(active_model_path)
        if model_hash_before != model_hash_after:
            print("[INFO] Model berubah, sudah direload otomatis.")

    if 'images' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diupload'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'File kosong'}), 400

    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    result_folder = current_app.config.get('RESULT_FOLDER', 'static/results')

    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)

    uploaded_files = []

    model = get_model()

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(upload_folder, filename)
            file.save(upload_path)

            # Proses prediksi segmentasi dengan model
            results = model.predict(source=upload_path, imgsz=640, conf=0.25, verbose=False)

            # Ambil mask overlay hasil segmentasi frame pertama (biasanya hanya 1 result)
            img_result = results[0].plot()

            result_img_path = os.path.join(result_folder, filename)
            cv2.imwrite(result_img_path, img_result)

            uploaded_files.append({
                'filename': filename,
                'input_url': f'/static/uploads/{filename}',
                'result_url': f'/static/results/{filename}'
            })
        else:
            return jsonify({'error': f'Format file tidak didukung: {file.filename}'}), 400

    return jsonify({'uploaded_files': uploaded_files})


@segmentation_file_multi_bp.route('/download-zip', methods=['POST'])
def segmentation_file_multi_download_zip():
    files = request.json.get('files', [])
    if not files:
        return jsonify({'error': 'Tidak ada file yang dipilih untuk download'}), 400

    result_folder = current_app.config.get('RESULT_FOLDER', 'static/results')
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in files:
            safe_filename = secure_filename(filename)
            file_path = os.path.join(result_folder, safe_filename)
            if os.path.exists(file_path):
                zf.write(file_path, arcname=safe_filename)

    memory_file.seek(0)
    return send_file(memory_file, mimetype='application/zip', as_attachment=True, download_name='segmentation_results.zip')


@segmentation_file_multi_bp.route('/delete', methods=['POST'])
def segmentation_file_multi_delete():
    files = request.json.get('files', [])
    if not files:
        return jsonify({'error': 'Tidak ada file yang dipilih untuk dihapus'}), 400

    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    result_folder = current_app.config.get('RESULT_FOLDER', 'static/results')

    deleted = []
    not_found = []

    for filename in files:
        safe_filename = secure_filename(filename)
        upload_path = os.path.join(upload_folder, safe_filename)
        result_path = os.path.join(result_folder, safe_filename)

        removed_any = False
        if os.path.exists(upload_path):
            os.remove(upload_path)
            removed_any = True
        if os.path.exists(result_path):
            os.remove(result_path)
            removed_any = True

        if removed_any:
            deleted.append(safe_filename)
        else:
            not_found.append(safe_filename)

    return jsonify({'deleted': deleted, 'not_found': not_found})
