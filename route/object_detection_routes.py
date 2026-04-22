import logging
import os
import cv2
import base64
import numpy as np
from flask import Blueprint, render_template, request, current_app, url_for, flash, redirect, jsonify
from werkzeug.utils import secure_filename
from route.utils import mark_active_menu
from menu_config import menus
from route.model_manager import get_model, load_model  # import model manager

logger = logging.getLogger(__name__)

obj_detect_bp = Blueprint('object_detection', __name__, template_folder='../templates')

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

@obj_detect_bp.route('/reload-model')
def reload_model():
    try:
        load_model()  # reload model dari file aktif di model_manager
        flash('Model deteksi objek berhasil diaktifkan ulang.', 'success')
    except Exception as e:
        flash(f'Gagal reload model: {e}', 'danger')
    return redirect(url_for('main.home'))

@obj_detect_bp.route('/', methods=['GET', 'POST'])
def detect():
    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection.html', menus=menus_active, title="Object Detection")

@obj_detect_bp.route('/file', methods=['GET', 'POST'])
def detect_file():
    error = None
    input_image = None
    result_image = None
    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'Tidak ada file yang diupload'
        else:
            file = request.files['image']
            if file.filename == '':
                error = 'Nama file kosong'
            elif file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

                model = get_model()  # ambil model dari model_manager

                results = model(upload_path)

                img_result = results[0].plot()

                result_img_path = os.path.join(current_app.config['RESULT_FOLDER'], filename)
                cv2.imwrite(result_img_path, img_result)

                input_image = url_for('static', filename='uploads/' + filename)
                result_image = url_for('static', filename='results/' + filename)
            else:
                error = 'Format file tidak didukung'

    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection_file.html',
                           error=error,
                           input_image=input_image,
                           result_image=result_image,
                           menus=menus_active,
                           title="Object Detection - File")

@obj_detect_bp.route('/camera')
def detect_camera():
    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection_camera.html', menus=menus_active, title="Object Detection - Camera")

@obj_detect_bp.route('/video')
def detect_video():
    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection_video.html', menus=menus_active, title="Object Detection - Video")

@obj_detect_bp.route('/api/detect-frame', methods=['POST'])
def detect_frame_api():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image dari frontend
        img_data = data['image'].split(',')[1]  # buang prefix "data:image/jpeg;base64,"
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Resize ke 320x240 untuk inferensi cepat
        img_resized = cv2.resize(img, (320, 240))

        model = get_model()  # dapatkan model yang aktif

        # Jalankan prediksi dengan device yang sesuai
        results = model.predict(source=img_resized, imgsz=320, conf=0.25, verbose=False)

        boxes = []
        masks = []

        scale_x = img.shape[1] / 320
        scale_y = img.shape[0] / 240

        for result in results:
            # Loop semua objek terdeteksi di frame (biasanya satu result per frame)
            for box in result.boxes.data.cpu().numpy():
                x1, y1, x2, y2, score, cls = box
                boxes.append({
                    'x': float(x1 * scale_x),
                    'y': float(y1 * scale_y),
                    'width': float((x2 - x1) * scale_x),
                    'height': float((y2 - y1) * scale_y),
                    'confidence': float(score),
                    'label': model.names[int(cls)]
                })

            # Ambil mask setiap objek, encode ke base64 PNG jika ada
            if hasattr(result, 'masks') and result.masks is not None:
                masks_tensor = result.masks.data.cpu().numpy()  # (N, H, W) boolean array
                for mask_arr in masks_tensor:
                    mask_img = (mask_arr * 255).astype(np.uint8)
                    ret, buf = cv2.imencode('.png', mask_img)
                    if ret:
                        encoded_mask = base64.b64encode(buf).decode('utf-8')
                        masks.append(f"data:image/png;base64,{encoded_mask}")

        return jsonify({
            'boxes': boxes,
            'masks': masks
        })

    except Exception as e:
        logger.error('detect_frame_api: %s', e)
        return jsonify({'error': 'Internal server error'}), 500
