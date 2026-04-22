import os
import cv2
import numpy as np
import random
from flask import Blueprint, request, render_template, current_app, flash, redirect, jsonify, send_file
from werkzeug.utils import secure_filename
from route.model_manager import get_model, load_model
from route.utils import mark_active_menu
from menu_config import menus
from skimage import measure
import yaml

from shapely.geometry import Polygon
from shapely.ops import unary_union

experimental_sahi_bp = Blueprint('experimental_sahi', __name__, template_folder='../templates')

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


def allowed_file(filename, allowed_exts):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_exts


def slice_image(image, slice_size=512, overlap=0.2):
    h, w = image.shape[:2]
    step = int(slice_size * (1 - overlap))
    slices = []
    coords = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            x_end = min(x + slice_size, w)
            y_end = min(y + slice_size, h)
            patch = image[y:y_end, x:x_end]
            slices.append(patch)
            coords.append((x, y, x_end, y_end))
            if x_end == w:
                break
        if y_end == h:
            break
    return slices, coords


def stitch_mask_to_full(mask, coord, full_shape):
    x1, y1, x2, y2 = coord
    full_mask = np.zeros(full_shape[:2], dtype=np.uint8)
    h, w = y2 - y1, x2 - x1
    resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    full_mask[y1:y2, x1:x2] = resized_mask
    return full_mask.astype(bool)


def save_yolo_segmentation_txt(filepath, masks, class_ids, class_id_map, image_width, image_height):
    with open(filepath, 'w') as f:
        for cls_id_orig, mask in zip(class_ids, masks):
            cls_id = class_id_map[cls_id_orig]
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
                f.write(f"{cls_id} {coords_str}\n")


def choose_dataset_split(train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    r = random.random()
    if r < train_ratio:
        return 'train'
    elif r < train_ratio + val_ratio:
        return 'val'
    else:
        return 'test'


def generate_yolo_dataset_yaml(base_dataset_folder, used_class_ids, model_names, train_path='dataset/train/images', val_path='dataset/val/images', test_path='dataset/test/images', task='segment'):
    used_class_names = [model_names[i] for i in used_class_ids]
    names = {i: name for i, name in enumerate(used_class_names)}
    data = {
        'train': f'../{train_path}',
        'val': f'../{val_path}',
        'test': f'../{test_path}',
        'nc': len(used_class_names),
        'names': names,
        'task': task
    }
    yaml_path = os.path.join(base_dataset_folder, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'[INFO] Dataset YAML dibuat dengan kelas terpakai di {yaml_path}')
    return yaml_path


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if len(cnt) >= 3:
            poly = Polygon(cnt.squeeze(axis=1))
            if poly.is_valid and poly.area > 10:
                polygons.append(poly)
    return polygons


def merge_polygons(polygons):
    if not polygons:
        return []
    merged = unary_union(polygons)
    if merged.geom_type == 'Polygon':
        return [merged]
    elif merged.geom_type == 'MultiPolygon':
        return list(merged.geoms)
    else:
        return []


def polygon_nms(polygons, scores, iou_threshold=0.5):
    """
    Melakukan Non-Maximum Suppression pada daftar poligon berdasarkan skor dan IoU.
    
    Args:
        polygons (list of Polygon): Daftar poligon Shapely.
        scores (list of float): Skor kepercayaan untuk setiap poligon.
        iou_threshold (float): Ambang batas IoU untuk menganggap dua poligon tumpang tindih.
    
    Returns:
        list of int: Indeks poligon yang dipertahankan setelah NMS.
    """
    if not polygons:
        return []
    
    # Urutkan poligon berdasarkan skor (dari tertinggi ke terendah)
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    keep = []
    
    while sorted_indices:
        # Pilih poligon dengan skor tertinggi
        i = sorted_indices.pop(0)
        keep.append(i)
        
        # Hitung IoU dengan poligon yang tersisa
        current_polygon = polygons[i]
        remaining_polygons = [polygons[j] for j in sorted_indices]
        ious = []
        for p in remaining_polygons:
            try:
                iou = current_polygon.intersection(p).area / current_polygon.union(p).area
            except:
                iou = 0.0
            ious.append(iou)
        
        # Hapus poligon dengan IoU di atas ambang batas
        sorted_indices = [j for j, iou in zip(sorted_indices, ious) if iou < iou_threshold]
    
    return keep


@experimental_sahi_bp.route('/experimental/auto-annotation-sahi', methods=['GET', 'POST'])
def auto_annotation_sahi():
    menus_active = mark_active_menu(request.path, menus)

    if request.method == 'POST':
        active_model_file = os.path.join(os.path.dirname(__file__), '..', 'models', 'active_model.txt')
        if not os.path.exists(active_model_file):
            flash('Model aktif tidak ditemukan.', 'danger')
            return redirect(request.url)
        with open(active_model_file, 'r') as f:
            active_model_path = f.read().strip()

        try:
            load_model()
        except Exception as e:
            flash(f'Gagal load model: {e}', 'danger')
            return redirect(request.url)

        model = get_model()
        model_names = model.names

        try:
            train_ratio = float(request.form.get('train_ratio', 0.7))
            val_ratio = float(request.form.get('val_ratio', 0.2))
            test_ratio = float(request.form.get('test_ratio', 0.1))
            total = train_ratio + val_ratio + test_ratio
            if abs(total - 1.0) > 1e-3:
                flash('Total rasio train+val+test harus 1.0', 'danger')
                return redirect(request.url)
        except Exception:
            flash('Rasio split tidak valid.', 'danger')
            return redirect(request.url)

        fps_sample = request.form.get('fps_sample', '2')
        try:
            fps_sample = float(fps_sample)
            if fps_sample <= 0:
                fps_sample = 2
        except:
            fps_sample = 2

        enable_merge = request.form.get('enable_polygon_merge') == '1'
        enable_nms = request.form.get('enable_polygon_nms') == '1'
        iou_threshold = float(request.form.get('iou_threshold', 0.5))

        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
        base_dataset_folder = os.path.join(current_app.root_path, 'static', 'dataset')

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(base_dataset_folder, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(base_dataset_folder, split, 'labels'), exist_ok=True)

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

        used_class_ids_set = set()

        if ext in ALLOWED_IMAGE_EXTENSIONS:
            img = cv2.imread(filepath)
            h, w = img.shape[:2]
            patches, coords = slice_image(img, slice_size=512, overlap=0.2)

            all_masks = []
            all_class_ids = []

            for patch, coord in zip(patches, coords):
                results = model(patch)
                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    for mask, cls_id in zip(masks, classes):
                        full_mask = stitch_mask_to_full(mask, coord, img.shape)
                        all_masks.append(full_mask)
                        all_class_ids.append(cls_id)
                        used_class_ids_set.add(cls_id)

            # Apply NMS to polygons
            if enable_merge:
                merged_masks = []
                merged_class_ids = []
                polygons_per_class = {}

                for mask, cls_id in zip(all_masks, all_class_ids):
                    polys = mask_to_polygons(mask)
                    polygons_per_class.setdefault(cls_id, []).extend(polys)

                for cls_id, polys in polygons_per_class.items():
                    # Apply NMS
                    if enable_nms:
                        # Example scores - replace with actual confidence scores from your model
                        scores = [0.9] * len(polys)
                        keep_indices = polygon_nms(polys, scores, iou_threshold=iou_threshold)
                        filtered_polys = [polys[i] for i in keep_indices]
                    else:
                        filtered_polys = polys

                    merged_polys = merge_polygons(filtered_polys)
                    for merged_poly in merged_polys:
                        mask_np = np.zeros(all_masks[0].shape, dtype=np.uint8)
                        exterior_coords = np.array(merged_poly.exterior.coords).round().astype(np.int32)
                        cv2.fillPoly(mask_np, [exterior_coords], 1)
                        merged_masks.append(mask_np.astype(bool))
                        merged_class_ids.append(cls_id)

                all_masks = merged_masks
                all_class_ids = merged_class_ids

            used_class_ids = sorted(used_class_ids_set)
            class_id_map = {orig: new_idx for new_idx, orig in enumerate(used_class_ids)}

            split = choose_dataset_split(train_ratio, val_ratio, test_ratio)
            image_folder = os.path.join(base_dataset_folder, split, 'images')
            label_folder = os.path.join(base_dataset_folder, split, 'labels')

            save_image_path = os.path.join(image_folder, filename)
            cv2.imwrite(save_image_path, img)

            label_save_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')
            save_yolo_segmentation_txt(label_save_path, all_masks, all_class_ids, class_id_map, w, h)

            generate_yolo_dataset_yaml(base_dataset_folder, used_class_ids, model_names, task='segment')

            message = f'Gambar berhasil di-segmentasi dan dataset disimpan ke folder {split}.'
            annotation_files = [os.path.relpath(save_image_path, base_dataset_folder),
                                os.path.relpath(label_save_path, base_dataset_folder)]

            return render_template('experimental_auto_annotation_result.html',
                                   menus=menus_active,
                                   title="Hasil Auto Annotation Gambar",
                                   message=message,
                                   annotation_files=annotation_files)

        else:
            cap = cv2.VideoCapture(filepath)
            video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            sample_interval = max(1, int(round(video_fps / fps_sample)))

            frame_idx = 0
            saved_frame_idx = 0
            annotation_files = []
            # Tunda penulisan label sampai semua frame diproses agar class_id_map konsisten
            pending_labels = []

            video_base_name = os.path.splitext(filename)[0]

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    h, w = frame.shape[:2]
                    patches, coords = slice_image(frame, slice_size=512, overlap=0.2)

                    all_masks = []
                    all_class_ids = []

                    for patch, coord in zip(patches, coords):
                        results = model(patch)
                        if results[0].masks is not None:
                            masks = results[0].masks.data.cpu().numpy()
                            classes = results[0].boxes.cls.cpu().numpy().astype(int)
                            for mask, cls_id in zip(masks, classes):
                                full_mask = stitch_mask_to_full(mask, coord, frame.shape)
                                all_masks.append(full_mask)
                                all_class_ids.append(cls_id)
                                used_class_ids_set.add(cls_id)

                    if enable_merge:
                        merged_masks = []
                        merged_class_ids = []
                        polygons_per_class = {}

                        for mask, cls_id in zip(all_masks, all_class_ids):
                            polys = mask_to_polygons(mask)
                            polygons_per_class.setdefault(cls_id, []).extend(polys)

                        for cls_id, polys in polygons_per_class.items():
                            if enable_nms:
                                scores = [0.9] * len(polys)
                                keep_indices = polygon_nms(polys, scores, iou_threshold=iou_threshold)
                                filtered_polys = [polys[i] for i in keep_indices]
                            else:
                                filtered_polys = polys

                            merged_polys = merge_polygons(filtered_polys)
                            ref_shape = all_masks[0].shape if all_masks else frame.shape[:2]
                            for merged_poly in merged_polys:
                                mask_np = np.zeros(ref_shape, dtype=np.uint8)
                                exterior_coords = np.array(merged_poly.exterior.coords).round().astype(np.int32)
                                cv2.fillPoly(mask_np, [exterior_coords], 1)
                                merged_masks.append(mask_np.astype(bool))
                                merged_class_ids.append(cls_id)

                        all_masks = merged_masks
                        all_class_ids = merged_class_ids

                    split = choose_dataset_split(train_ratio, val_ratio, test_ratio)
                    image_folder = os.path.join(base_dataset_folder, split, 'images')
                    label_folder = os.path.join(base_dataset_folder, split, 'labels')

                    os.makedirs(image_folder, exist_ok=True)
                    os.makedirs(label_folder, exist_ok=True)

                    frame_filename = f'{video_base_name}_frame_{saved_frame_idx:05d}.png'
                    label_filename = f'{video_base_name}_frame_{saved_frame_idx:05d}.txt'

                    frame_save_path = os.path.join(image_folder, frame_filename)
                    label_save_path = os.path.join(label_folder, label_filename)

                    cv2.imwrite(frame_save_path, frame)
                    # Simpan data label untuk ditulis setelah semua frame diproses
                    pending_labels.append((label_save_path, all_masks, all_class_ids, w, h))

                    annotation_files.append(os.path.relpath(frame_save_path, base_dataset_folder))
                    annotation_files.append(os.path.relpath(label_save_path, base_dataset_folder))

                    saved_frame_idx += 1

                frame_idx += 1

            cap.release()

            # Buat class_id_map final dari seluruh kelas yang muncul di semua frame
            final_used_class_ids = sorted(used_class_ids_set)
            final_class_id_map = {orig: new_idx for new_idx, orig in enumerate(final_used_class_ids)}

            for lpath, masks, class_ids, fw, fh in pending_labels:
                save_yolo_segmentation_txt(lpath, masks, class_ids, final_class_id_map, fw, fh)

            generate_yolo_dataset_yaml(base_dataset_folder, final_used_class_ids, model_names, task='segment')

            message = f'Video berhasil diproses dan dataset disimpan dengan {saved_frame_idx} frame.'
            return render_template('experimental_auto_annotation_result.html',
                                   menus=menus_active,
                                   title="Hasil Auto Annotation Video",
                                   message=message,
                                   annotation_files=annotation_files)

    # GET method
    return render_template('experimental_auto_annotation_sahi.html',
                           menus=menus_active,
                           title="Auto Annotation - Manual Slicing Segmentasi")