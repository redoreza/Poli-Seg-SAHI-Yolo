from flask import Blueprint, render_template, request, jsonify
from route.utils import mark_active_menu
from menu_config import menus
import cv2
import numpy as np
import base64
from route.model_manager import get_model

bp = Blueprint('object_tracker_segmentation', __name__, template_folder='../templates')

@bp.route('/object-tracker-segmentation')
def object_tracker_segmentation_page():
    menus_active = mark_active_menu(request.path, menus)
    model_path = getattr(get_model(), 'model_path', 'Model belum dimuat')
    return render_template(
        'object_tracker_segmentation.html',
        menus=menus_active,
        title="Object Tracker Segmentation",
        model_path=model_path,
    )

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        idxs = idxs[1:]
        idxs = np.array([i for i in idxs if iou(boxes[current], boxes[i]) < iou_threshold])
    return keep

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

@bp.route('/api/track-segmentation', methods=['POST'])
def api_track_segmentation():
    data = request.get_json()
    if not data or 'image' not in data or 'enable_segmentation' not in data:
        return jsonify({'error': 'Data tidak lengkap'}), 400

    img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    enable_segmentation = bool(data['enable_segmentation'])

    model = get_model()
    if model is None:
        return jsonify({'error': 'Model belum dimuat'}), 500

    patches, coords = slice_image(frame, slice_size=512, overlap=0.2)

    all_boxes_data = []
    all_masks_data = []

    for patch, (x1, y1, x2, y2) in zip(patches, coords):
        results = model.track(source=patch, persist=True, imgsz=320, conf=0.3)

        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else np.arange(len(boxes))
            classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.array([])

            for i, (box, obj_id, cls) in enumerate(zip(boxes, ids, classes)):
                bx1, by1, bx2, by2, conf, _ = box[:6]
                label = model.names[int(cls)] if hasattr(model, 'names') else str(int(cls))

                all_boxes_data.append({
                    'x': float(bx1 + x1),
                    'y': float(by1 + y1),
                    'width': float(bx2 - bx1),
                    'height': float(by2 - by1),
                    'confidence': float(conf),
                    'label': label,
                    'id': int(obj_id),
                })

                if enable_segmentation and hasattr(result, 'masks') and result.masks is not None:
                    mask = result.masks.data[i].cpu().numpy()
                    _, mask_img = cv2.imencode('.png', (mask * 255).astype(np.uint8))
                    mask_b64 = base64.b64encode(mask_img).decode('utf-8')

                    all_masks_data.append({
                        'mask_b64': f"data:image/png;base64,{mask_b64}",
                        'x1': x1,
                        'y1': y1,
                        'width': x2 - x1,
                        'height': y2 - y1
                    })

    if all_boxes_data:
        boxes_array = np.array([[b['x'], b['y'], b['width'], b['height']] for b in all_boxes_data])
        scores = np.array([b['confidence'] for b in all_boxes_data])
        keep_indices = nms(boxes_array, scores, iou_threshold=0.5)
        filtered_boxes = [all_boxes_data[i] for i in keep_indices]
    else:
        filtered_boxes = []

    return jsonify({
        'boxes': filtered_boxes,
        'masks': all_masks_data if enable_segmentation else [],
        'model_path': getattr(model, 'model_path', ''),
    })
