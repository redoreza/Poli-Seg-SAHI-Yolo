import logging
import os
import cv2
import base64
import numpy as np
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from menu_config import menus
from route.utils import mark_active_menu
from route.model_manager import get_model, load_model  # import model manager

logger = logging.getLogger(__name__)

segmentation_bp = Blueprint('segmentation', __name__, template_folder='../templates')

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

@segmentation_bp.route('/reload-model')
def reload_model():
    try:
        load_model()  # reload model dari file aktif di model_manager
        flash('Model segmentasi berhasil diaktifkan ulang.', 'success')
    except Exception as e:
        flash(f'Gagal reload model: {e}', 'danger')
    return redirect(url_for('main.home'))

@segmentation_bp.route('/camera')
def segmentation_camera():
    menus_active = mark_active_menu(request.path, menus)
    return render_template('segmentation_camera.html', menus=menus_active, title="Segmentation Camera")

@segmentation_bp.route('/api/detect-frame', methods=['POST'])
def detect_frame_api():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode base64 image dari frontend
        img_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Resize ke 320x240 untuk inferensi cepat
        img_resized = cv2.resize(img, (320, 240))
        scale_x = img.shape[1] / 320
        scale_y = img.shape[0] / 240

        model = get_model()  # ambil model dari model_manager

        results = model.predict(source=img_resized, imgsz=320, conf=0.25, verbose=False)

        boxes = []
        masks = []

        for result in results:
            # Ambil semua objek terdeteksi
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

            # Ambil mask setiap objek, encode ke base64 PNG
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
        logger.error('detect_frame_api segmentation: %s', e)
        return jsonify({'error': 'Internal server error'}), 500
