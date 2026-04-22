import os
import cv2
import numpy as np
import json
import base64
import time
import zipfile
import io
import random
import yaml
from flask import Blueprint, request, render_template, current_app, jsonify, send_file, flash, redirect, url_for
from route.utils import mark_active_menu
from menu_config import menus
from route.model_manager import get_model, load_model
from werkzeug.utils import secure_filename
from skimage import measure

experimental_bp = Blueprint('experimental', __name__, template_folder='../templates')

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts

def choose_dataset_split(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    r = random.random()
    if r < train_ratio:
        return 'train'
    elif r < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'

def save_yolo_annotation_txt(filepath, boxes, image_width, image_height):
    with open(filepath, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            x_center = ((x1 + x2) / 2) / image_width
            y_center = ((y1 + y2) / 2) / image_height
            w = (x2 - x1) / image_width
            h = (y2 - y1) / image_height
            f.write(f"{int(cls)} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

def save_yolo_segmentation_txt(filepath, masks, class_ids, image_width, image_height):
    with open(filepath, 'w') as f:
        for cls_id, mask in zip(class_ids, masks):
            contours = measure.find_contours(mask.astype(np.uint8), 0.5)
            for contour in contours:
                if len(contour) < 3:
                    continue
                coords = []
                for point in contour:
                    y, x = point
                    coords.append(x / image_width)
                    coords.append(y / image_height)
                coords_str = ' '.join(f'{c:.6f}' for c in coords)
                f.write(f"{int(cls_id)} {coords_str}\n")

def generate_yolo_dataset_yaml_v2(
    base_dataset_folder: str,
    model_names,
    train_path: str = 'dataset/train/images',
    val_path: str = 'dataset/val/images',
    test_path: str = 'dataset/test/images',
    workspace: str = 'your-workspace',
    project: str = 'your-project',
    version: int = 1,
    license_str: str = 'CC BY 4.0',
    url: str = '',
    task: str = 'segment'
) -> str:
    if isinstance(model_names, dict):
        names = {int(k): v for k, v in model_names.items()}
    elif isinstance(model_names, (list, tuple)):
        names = {i: name for i, name in enumerate(model_names)}
    else:
        raise ValueError("model_names must be list or dict")

    data = {
        'train': f'../{train_path}',
        'val': f'../{val_path}',
        'test': f'../{test_path}' if test_path else '',
        'nc': len(names),
        'names': names,
        'task': task,
        'roboflow': {
            'workspace': workspace,
            'project': project,
            'version': version,
            'license': license_str,
            'url': url
        }
    }

    if not data['test']:
        data.pop('test')

    os.makedirs(base_dataset_folder, exist_ok=True)
    yaml_path = os.path.join(base_dataset_folder, 'dataset.yaml')

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f'[INFO] YOLO dataset YAML generated at: {yaml_path}')
    return yaml_path

@experimental_bp.route('/experimental/auto-annotation', methods=['GET', 'POST'])
def auto_annotation():
    menus_active = mark_active_menu(request.path, menus)

    if request.method == 'POST':
        try:
            train_ratio = float(request.form.get('train_ratio', 0.7))
            val_ratio = float(request.form.get('val_ratio', 0.2))
            test_ratio = float(request.form.get('test_ratio', 0.1))
        except Exception:
            train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio == 0:
            train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        else:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        mode = request.form.get('mode')
        annotation_type = request.form.get('annotation_type')

        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        base_dataset_folder = os.path.join(current_app.root_path, 'static', 'dataset')
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(base_dataset_folder, exist_ok=True)

        try:
            load_model()
        except Exception as e:
            flash(f'Gagal load model: {e}', 'danger')
            return redirect(url_for('experimental.auto_annotation'))

        model = get_model()
        class_names = model.names

        task_type = 'segment' if annotation_type == 'segmentation' else 'detect'

        generate_yolo_dataset_yaml_v2(
            base_dataset_folder=base_dataset_folder,
            model_names=class_names,
            train_path='dataset/train/images',
            val_path='dataset/val/images',
            test_path='dataset/test/images',
            workspace='your-workspace',
            project='your-project',
            version=1,
            license_str='CC BY 4.0',
            url='',
            task=task_type
        )

        if mode == 'image_or_video':
            if 'file' not in request.files:
                flash('Tidak ada file yang diupload.', 'danger')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('Nama file kosong.', 'danger')
                return redirect(request.url)

            if '.' not in file.filename:
                flash('Format file tidak didukung.', 'danger')
                return redirect(request.url)
            ext = file.filename.rsplit('.', 1)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS.union(ALLOWED_VIDEO_EXTENSIONS):
                flash('Format file tidak didukung.', 'danger')
                return redirect(request.url)

            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)

            if ext in ALLOWED_IMAGE_EXTENSIONS:
                # Prediksi untuk file gambar
                results = model.predict(source=filepath, imgsz=640, conf=0.25, verbose=False)
                img = cv2.imread(filepath)
                h, w = img.shape[:2]
                filename_base = os.path.splitext(filename)[0]

                split_folder = choose_dataset_split(train_ratio, val_ratio, test_ratio)

                image_folder = os.path.join(base_dataset_folder, split_folder, 'images')
                label_folder = os.path.join(base_dataset_folder, split_folder, 'labels')
                os.makedirs(image_folder, exist_ok=True)
                os.makedirs(label_folder, exist_ok=True)

                img_save_path = os.path.join(image_folder, filename_base + '.png')
                cv2.imwrite(img_save_path, img)

                if annotation_type == 'detection':
                    boxes = results[0].boxes.data.cpu().numpy()
                    save_yolo_annotation_txt(label_save_path := os.path.join(label_folder, filename_base + '.txt'),
                                             boxes, w, h)
                    message = 'Dataset deteksi objek berhasil disimpan.'
                    annotation_files = [os.path.relpath(img_save_path, base_dataset_folder),
                                        os.path.relpath(label_save_path, base_dataset_folder)]

                else:
                    masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
                    if masks is None or len(masks) == 0:
                        flash('Model tidak menghasilkan segmentasi mask.', 'danger')
                        return redirect(request.url)

                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    label_save_path = os.path.join(label_folder, filename_base + '.txt')
                    save_yolo_segmentation_txt(label_save_path, masks, class_ids, w, h)
                    message = 'Label segmentasi polygon berhasil disimpan tanpa simpan file mask.'
                    annotation_files = [os.path.relpath(label_save_path, base_dataset_folder)]

                return render_template('experimental_auto_annotation_result.html',
                                       menus=menus_active,
                                       title="Hasil Simpan Dataset",
                                       message=message,
                                       annotation_files=annotation_files)

            else:
                # Proses untuk file video, dengan perbaikan skala anotasi
                cap = cv2.VideoCapture(filepath)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                filename_base = os.path.splitext(filename)[0]

                target_size = 640  # Ukuran input model YOLO

                frame_skip = max(1, int(round(fps / 1)))  # 1 frame per detik

                frame_idx = 0
                saved_frame_idx = 0
                all_files = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip == 0:
                        # Resize frame ke ukuran input model
                        img_resized = cv2.resize(frame, (target_size, target_size))

                        results = model.predict(source=img_resized, imgsz=target_size, conf=0.25, verbose=False)

                        split_folder = choose_dataset_split(train_ratio, val_ratio, test_ratio)

                        image_folder = os.path.join(base_dataset_folder, split_folder, 'images')
                        label_folder = os.path.join(base_dataset_folder, split_folder, 'labels')
                        os.makedirs(image_folder, exist_ok=True)
                        os.makedirs(label_folder, exist_ok=True)

                        frame_filename = f'{filename_base}_frame_{saved_frame_idx:05d}.png'
                        label_filename = f'{filename_base}_frame_{saved_frame_idx:05d}.txt'

                        frame_save_path = os.path.join(image_folder, frame_filename)
                        label_save_path = os.path.join(label_folder, label_filename)

                        # Hitung faktor skala dari model input ke ukuran frame asli
                        scale_x = width / target_size
                        scale_y = height / target_size

                        if annotation_type == 'detection':
                            boxes = results[0].boxes.data.cpu().numpy().copy()
                            # Skala ulang koordinat bounding box ke ukuran asli frame video
                            boxes[:, 0] *= scale_x  # x1
                            boxes[:, 2] *= scale_x  # x2
                            boxes[:, 1] *= scale_y  # y1
                            boxes[:, 3] *= scale_y  # y2

                            save_yolo_annotation_txt(label_save_path, boxes, width, height)

                        else:
                            masks = results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
                            if masks is not None and len(masks) > 0:
                                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                                masks_resized = []
                                for mask in masks:
                                    mask_uint8 = (mask * 255).astype('uint8')
                                    # Resize mask ke ukuran frame asli
                                    mask_resized = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
                                    masks_resized.append(mask_resized > 127)  # threshold kembali ke boolean
                                masks_resized = np.array(masks_resized)

                                save_yolo_segmentation_txt(label_save_path, masks_resized, class_ids, width, height)

                        # Simpan frame asli (bukan yang sudah di-resize)
                        cv2.imwrite(frame_save_path, frame)

                        all_files.append(os.path.relpath(frame_save_path, base_dataset_folder))
                        all_files.append(os.path.relpath(label_save_path, base_dataset_folder))

                        saved_frame_idx += 1

                    frame_idx += 1

                cap.release()
                message = f'Dataset video berhasil disimpan dengan {saved_frame_idx} frame.'

                return render_template('experimental_auto_annotation_result.html',
                                       menus=menus_active,
                                       title="Hasil Simpan Dataset Video",
                                       message=message,
                                       annotation_files=all_files)

        else:
            flash('Mode tidak valid.', 'danger')
            return redirect(request.url)

    # GET method: render halaman form
    return render_template('experimental_auto_annotation.html',
                           menus=menus_active,
                           title="Auto Annotation - Experimental")


@experimental_bp.route('/experimental/api/realtime-annotation', methods=['POST'])
def realtime_annotation_api():
    data = request.get_json()
    if not data or 'image' not in data or 'annotation_type' not in data:
        return jsonify({'error': 'Data tidak lengkap'}), 400

    try:
        load_model()
    except Exception as e:
        return jsonify({'error': f'Gagal load model: {e}'}), 500

    model = get_model()
    class_names = model.names

    img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Gambar tidak valid'}), 400

    annotation_type = data['annotation_type']

    results = model.predict(source=img, imgsz=320, conf=0.25, verbose=False)

    annotation_folder = os.path.join(current_app.root_path, 'static', 'annotations', 'realtime_session')
    os.makedirs(annotation_folder, exist_ok=True)

    timestamp = int(time.time() * 1000)

    if annotation_type == 'detection':
        boxes = results[0].boxes.data.cpu().numpy()
        annotation_path = os.path.join(annotation_folder, f'{timestamp}.txt')
        save_yolo_annotation_txt(annotation_path, boxes, img.shape[1], img.shape[0])

        boxes_list = []
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            boxes_list.append({
                'x1': float(x1),
                'y1': float(y1),
                'x2': float(x2),
                'y2': float(y2),
                'confidence': float(conf),
                'class_id': int(cls),
                'label': class_names[int(cls)] if int(cls) < len(class_names) else f'class_{int(cls)}'
            })
        return jsonify({'boxes': boxes_list})

    elif annotation_type == 'segmentation':
        masks_encoded = []
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            poly_path = os.path.join(annotation_folder, f'{timestamp}_poly.json')
            # Simpan polygon mask ke file JSON
            with open(poly_path, 'w') as f:
                polygons = []
                for mask in masks:
                    contours = measure.find_contours(mask.astype(np.uint8), 0.5)
                    poly_list = [contour.flatten().tolist() for contour in contours if len(contour) >= 3]
                    polygons.append(poly_list)
                json.dump(polygons, f)

            for mask in masks:
                mask_img = (mask * 255).astype(np.uint8)
                ret, buf = cv2.imencode('.png', mask_img)
                if ret:
                    mask_b64 = base64.b64encode(buf).decode('utf-8')
                    masks_encoded.append(f"data:image/png;base64,{mask_b64}")

        return jsonify({'masks': masks_encoded})

    else:
        return jsonify({'error': 'Jenis anotasi tidak dikenali'}), 400


@experimental_bp.route('/experimental/download-annotation-zip/<video_folder>')
def download_annotation_zip(video_folder):
    safe_base = os.path.realpath(os.path.join(current_app.root_path, 'static', 'annotations'))
    annotation_folder = os.path.realpath(os.path.join(safe_base, video_folder))
    if not annotation_folder.startswith(safe_base + os.sep):
        return "Path tidak valid", 400
    if not os.path.exists(annotation_folder):
        return "Folder anotasi tidak ditemukan", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(annotation_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, annotation_folder)
                zf.write(file_path, arcname=arcname)
    memory_file.seek(0)
    return send_file(memory_file,
                     mimetype='application/zip',
                     as_attachment=True,
                     download_name=f'{video_folder}_annotations.zip')


@experimental_bp.route('/experimental/download-dataset-zip')
def download_dataset_zip():
    dataset_folder = os.path.join(current_app.root_path, 'static', 'dataset')
    if not os.path.exists(dataset_folder):
        return "Folder dataset tidak ditemukan", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dataset_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_folder)
                zf.write(file_path, arcname=arcname)
    memory_file.seek(0)
    return send_file(memory_file,
                     mimetype='application/zip',
                     as_attachment=True,
                     download_name='dataset.zip')
