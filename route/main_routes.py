from flask import Blueprint, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from route.utils import mark_active_menu
from route.model_manager import load_model, clear_model_cache, get_model

main_bp = Blueprint('main', __name__)

MODEL_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'models')
ACTIVE_MODEL_FILE = os.path.join(MODEL_FOLDER, 'active_model.txt')

ALLOWED_MODEL_EXTENSIONS = {'pt'}

def allowed_model_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS

def get_model_list():
    try:
        files = [f for f in os.listdir(MODEL_FOLDER) if allowed_model_file(f)]
    except FileNotFoundError:
        files = []
    return files

@main_bp.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'model_file' not in request.files:
            flash('Tidak ada file yang diupload', 'danger')
            return redirect(request.url)

        file = request.files['model_file']
        if file.filename == '':
            flash('Nama file kosong', 'danger')
            return redirect(request.url)

        if file and allowed_model_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(MODEL_FOLDER, exist_ok=True)
            save_path = os.path.join(MODEL_FOLDER, filename)
            file.save(save_path)

            with open(ACTIVE_MODEL_FILE, 'w') as f:
                f.write(save_path)

            try:
                clear_model_cache()  # Hapus cache model lama
                load_model()         # Load model baru
            except Exception as e:
                flash(f'Gagal reload model: {e}', 'danger')
                return redirect(request.url)

            flash('Model berhasil diupload dan otomatis diaktifkan untuk semua fitur.', 'success')
            return redirect(url_for('main.home'))
        else:
            flash('Format file model tidak didukung. Harus berekstensi .pt', 'danger')
            return redirect(request.url)

    model_files = get_model_list()

    active_model_path = None
    if os.path.exists(ACTIVE_MODEL_FILE):
        with open(ACTIVE_MODEL_FILE, 'r') as f:
            active_model_path = f.read().strip()
    active_model_name = os.path.basename(active_model_path) if active_model_path else None

    # Dapatkan model aktif dan class labels
    model = get_model()
    class_names = getattr(model, 'names', None)

    if class_names:
        if isinstance(class_names, dict):
            class_labels = [class_names[k] for k in sorted(class_names.keys())]
        else:
            class_labels = list(class_names)
    else:
        class_labels = []

    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)

    return render_template('home.html',
                           menus=menus_active,
                           title="Home",
                           model_files=model_files,
                           active_model=active_model_name,
                           class_labels=class_labels)

@main_bp.route('/activate-model/<filename>')
def activate_model(filename):
    safe_filename = secure_filename(filename)
    model_path = os.path.join(MODEL_FOLDER, safe_filename)

    if not os.path.exists(model_path) or not allowed_model_file(safe_filename):
        flash('Model tidak ditemukan atau format tidak valid.', 'danger')
        return redirect(url_for('main.home'))

    with open(ACTIVE_MODEL_FILE, 'w') as f:
        f.write(model_path)

    try:
        clear_model_cache()  # Hapus cache model lama
        load_model()         # Load model baru
    except Exception as e:
        flash(f'Gagal reload model: {e}', 'danger')
        return redirect(url_for('main.home'))

    flash(f'Model "{safe_filename}" berhasil diaktifkan untuk semua fitur.', 'success')
    return redirect(url_for('main.home'))

@main_bp.route('/delete-model/<filename>')
def delete_model(filename):
    safe_filename = secure_filename(filename)
    model_path = os.path.join(MODEL_FOLDER, safe_filename)
    if os.path.exists(model_path) and allowed_model_file(safe_filename):
        try:
            os.remove(model_path)
            flash(f'Model "{safe_filename}" berhasil dihapus.', 'success')

            if os.path.exists(ACTIVE_MODEL_FILE):
                with open(ACTIVE_MODEL_FILE, 'r') as f:
                    active_path = f.read().strip()
                if active_path == model_path:
                    os.remove(ACTIVE_MODEL_FILE)
                    flash('Model aktif dihapus, silakan pilih model lain.', 'warning')
                    try:
                        clear_model_cache()  # Hapus cache model karena model aktif dihapus
                        load_model()         # Load default model atau tidak ada model
                    except Exception as e:
                        flash(f'Gagal reload model: {e}', 'danger')

        except Exception as e:
            flash(f'Gagal menghapus model: {e}', 'danger')
    else:
        flash('Model tidak ditemukan atau format tidak valid.', 'danger')

    return redirect(url_for('main.home'))

@main_bp.route('/under-construction')
def under_construction():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('under_construction.html', menus=menus_active, title="Dalam Proses")

@main_bp.route('/page2')
def page2():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('page2.html', menus=menus_active, title="Page 2")

@main_bp.route('/object-detection/camera')
def object_detection_camera():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection_camera.html', menus=menus_active, title="Object Detection - Camera")

@main_bp.route('/object-detection/video')
def object_detection_video():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('video_detection_upload.html', menus=menus_active, title="Object Detection - Video")

@main_bp.route('/object-detection/file-multi')
def object_detection_file_multi():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection_file_multi.html', menus=menus_active, title="Object Detection - Multi File Upload")

@main_bp.route('/object-detection')
def object_detection():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('object_detection.html', menus=menus_active, title="Object Detection")

@main_bp.route('/segmentation/camera')
def segmentation_camera():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('segmentation_camera.html', menus=menus_active, title="Segmentation Camera")

@main_bp.route('/setting/profile')
def setting_profile():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('setting_profile.html', menus=menus_active, title="Profile Setting")

@main_bp.route('/reload-active-model')
def reload_active_model():
    try:
        clear_model_cache()
        load_model()
        flash('Model aktif berhasil di-reload.', 'success')
    except Exception as e:
        flash(f'Gagal reload model: {e}', 'danger')
    return redirect(url_for('main.home'))

@main_bp.route('/segmentation/video')
def segmentation_video():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('segmentation_video_upload.html', menus=menus_active, title="Segmentation - Video Upload")

@main_bp.route('/segmentation/image')
def segmentation_image_multi():
    from menu_config import menus
    menus_active = mark_active_menu(request.path, menus)
    return render_template('segmentation_file_multi.html', menus=menus_active, title="Segmentation - Multi File Upload")
