from flask import Blueprint, request, render_template, redirect, url_for, flash, jsonify
from menu_config import menus
from route.utils import mark_active_menu
import os

setting_bp = Blueprint('setting', __name__, template_folder='../templates')

# Simpan setting sederhana dalam file (atau gunakan DB)
SETTING_FILE = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_setting.txt')

@setting_bp.route('/setting/camera', methods=['GET', 'POST'])
def setting_camera():
    selected_camera = None
    camera_width = 640
    camera_height = 480

    # Load existing settings jika ada
    if os.path.exists(SETTING_FILE):
        with open(SETTING_FILE, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 3:
                try:
                    selected_camera = lines[0]
                    camera_width = int(lines[1])
                    camera_height = int(lines[2])
                except (ValueError, IndexError):
                    selected_camera = None
                    camera_width = 640
                    camera_height = 480

    if request.method == 'POST':
        selected_camera = request.form.get('selected_camera')
        try:
            camera_width = int(request.form.get('camera_width', 640))
            camera_height = int(request.form.get('camera_height', 480))
        except ValueError:
            flash("Nilai lebar atau tinggi kamera tidak valid.", "danger")
            return redirect(url_for('setting.setting_camera'))

        # Simpan setting ke file (bisa disesuaikan simpan ke DB)
        os.makedirs(os.path.dirname(SETTING_FILE), exist_ok=True)  # pastikan folder ada
        with open(SETTING_FILE, 'w') as f:
            f.write(f"{selected_camera}\n{camera_width}\n{camera_height}")

        flash("Pengaturan kamera berhasil disimpan.", "success")
        return redirect(url_for('setting.setting_camera'))

    menus_active = mark_active_menu(request.path, menus)
    return render_template('setting_camera.html',
                           menus=menus_active,
                           title="Setting Kamera",
                           selected_camera=selected_camera,
                           camera_width=camera_width,
                           camera_height=camera_height)

# === Tambahan route API untuk ambil setting kamera dalam JSON ===
@setting_bp.route('/setting/api/get-camera-setting')
def get_camera_setting():
    selected_camera = None
    camera_width = 640
    camera_height = 480

    if os.path.exists(SETTING_FILE):
        with open(SETTING_FILE, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 3:
                try:
                    selected_camera = lines[0]
                    camera_width = int(lines[1])
                    camera_height = int(lines[2])
                except (ValueError, IndexError):
                    selected_camera = None
                    camera_width = 640
                    camera_height = 480

    return jsonify({
        'selected_camera': selected_camera,
        'camera_width': camera_width,
        'camera_height': camera_height
    })
