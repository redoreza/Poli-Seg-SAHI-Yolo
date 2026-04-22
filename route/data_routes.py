import os
import yaml
import cv2
from math import ceil
from flask import Blueprint, render_template, current_app, jsonify, request
from route.utils import mark_active_menu
from menu_config import menus

data_bp = Blueprint('data', __name__, url_prefix='/data')


def load_class_names_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        return {}
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        names = data.get('names', {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            return {i: name for i, name in enumerate(names)}
        else:
            return {}
    except Exception as e:
        print(f"[ERROR] Gagal baca file YAML: {e}")
        return {}


def parse_label_file(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:  # minimal 3 titik polygon (6 koordinat)
                    continue
                cls_id = int(parts[0])
                coords = list(map(float, parts[1:]))
                labels.append({'class_id': cls_id, 'coords': coords})
    except Exception as e:
        print(f"[ERROR] Gagal parse file label {label_path}: {e}")
    return labels


def scan_dataset_files_with_labels():
    dataset_root = os.path.join(current_app.root_path, 'static', 'dataset')
    dataset = []
    class_count = {}

    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(dataset_root, split, 'images')
        labels_dir = os.path.join(dataset_root, split, 'labels')

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue

        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            label_file_name = os.path.splitext(fname)[0] + '.txt'
            label_path_abs = os.path.join(labels_dir, label_file_name)
            labels = parse_label_file(label_path_abs)

            img_path_abs = os.path.join(images_dir, fname)
            img = cv2.imread(img_path_abs)
            if img is not None:
                h, w = img.shape[:2]
            else:
                h, w = 0, 0

            for label in labels:
                class_count[label['class_id']] = class_count.get(label['class_id'], 0) + 1

            image_path = os.path.join('dataset', split, 'images', fname).replace(os.path.sep, '/')
            label_path = os.path.join('dataset', split, 'labels', label_file_name).replace(os.path.sep, '/')

            dataset.append({
                'split': split,
                'image': image_path,
                'label': label_path,
                'labels': labels,
                'label_exists': os.path.exists(label_path_abs),
                'image_width': w,
                'image_height': h,
            })

    return dataset, class_count


@data_bp.route('/edit')
def edit_data():
    menus_active = mark_active_menu(request.path, menus)
    all_data, class_counts = scan_dataset_files_with_labels()

    # Ambil parameter pagination dari query params, default page=1, per_page=10
    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1
    try:
        per_page = int(request.args.get('per_page', 10))
    except ValueError:
        per_page = 10

    per_page = min(max(per_page, 1), 50)  # batasi per_page antara 1 dan 50

    total_items = len(all_data)
    total_pages = ceil(total_items / per_page)

    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    data_subset = all_data[start_idx:end_idx]

    yaml_path = os.path.join(current_app.root_path, 'static', 'dataset', 'dataset.yaml')
    class_names = load_class_names_from_yaml(yaml_path)

    return render_template('data_edit.html',
                           data=data_subset,
                           class_counts=class_counts,
                           class_names=class_names,
                           title='Edit Data Hasil Auto Annotation',
                           menus=menus_active,
                           page=page,
                           per_page=per_page,
                           total_pages=total_pages)


@data_bp.route('/edit/delete', methods=['POST'])
def delete_data():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Data JSON tidak ditemukan'}), 400

    image_path = data.get('image')
    label_path = data.get('label')

    if not image_path or not label_path:
        return jsonify({'error': 'Parameter "image" dan "label" wajib ada'}), 400

    safe_base = os.path.realpath(os.path.join(current_app.root_path, 'static', 'dataset'))
    abs_image_path = os.path.realpath(os.path.join(current_app.root_path, 'static', image_path))
    abs_label_path = os.path.realpath(os.path.join(current_app.root_path, 'static', label_path))

    if not abs_image_path.startswith(safe_base + os.sep) or not abs_label_path.startswith(safe_base + os.sep):
        return jsonify({'error': 'Path tidak valid'}), 400

    errors = []

    try:
        if os.path.isfile(abs_image_path):
            os.remove(abs_image_path)
        else:
            errors.append('File gambar tidak ditemukan')
    except Exception as e:
        errors.append(f'Gagal menghapus gambar: {e}')

    try:
        if os.path.isfile(abs_label_path):
            os.remove(abs_label_path)
        else:
            errors.append('File label tidak ditemukan')
    except Exception as e:
        errors.append(f'Gagal menghapus label: {e}')

    if errors:
        return jsonify({'error': '; '.join(errors)}), 500

    return jsonify({'success': True})
