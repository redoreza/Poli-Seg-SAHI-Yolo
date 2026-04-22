import logging
import os
import subprocess
import sys
import threading
import cv2
import numpy as np
import time
import base64
from flask import Blueprint, render_template, request, current_app, jsonify
from route.model_manager import get_model, load_model
from route.utils import mark_active_menu
from menu_config import menus
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

object_tracker_bp = Blueprint('object_tracker', __name__, template_folder='../templates')

SETTING_FILE  = os.path.join(os.path.dirname(__file__), '..', 'config', 'camera_setting.txt')
COOKIES_FILE  = os.path.join(os.path.dirname(__file__), '..', 'config', 'yt_cookies.txt')
CONFIG_DIR    = os.path.join(os.path.dirname(__file__), '..', 'config')

# ── Tracker state ────────────────────────────────────────────────────────────
tracker_state   = {}
TIMEOUT_SECONDS = 10
last_activity   = 0
_tracker_lock   = threading.Lock()

CLASS_NAMES    = {0: "person", 2: "car", 5: "bus", 7: "truck"}
ALLOWED_CLASSES = {0, 2, 5, 7}

# ── Geometry helpers ─────────────────────────────────────────────────────────

def _cross(px, py, line):
    """Signed cross product (B-A)×(P-A). Sign = which side of line."""
    ax, ay = line['x1'], line['y1']
    bx, by = line['x2'], line['y2']
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _sign(v):
    return 1 if v > 0 else (-1 if v < 0 else 0)


# ── Routes ───────────────────────────────────────────────────────────────────

@object_tracker_bp.route('/camera', methods=['GET'])
def object_tracker_camera():
    menus_active = mark_active_menu(request.path, menus)
    selected_camera, w, h = None, 640, 480

    if os.path.exists(SETTING_FILE):
        try:
            lines = open(SETTING_FILE).read().splitlines()
            if len(lines) >= 3:
                selected_camera = lines[0] if lines[0].isdigit() else None
                w, h = int(lines[1]), int(lines[2])
        except (ValueError, IndexError):
            pass

    try:
        load_model()
    except Exception as e:
        logger.error("Model load failed: %s", e)

    return render_template('object_tracker_camera.html',
                           menus=menus_active, title="Object Speed Tracker",
                           selected_camera=selected_camera,
                           camera_width=w, camera_height=h)


@object_tracker_bp.route('/api/model-status')
def model_status():
    from route.model_manager import current_model_path
    if not current_model_path:
        return jsonify({'status': 'none'})
    return jsonify({'status': 'loaded', 'model_name': os.path.basename(current_model_path)})


@object_tracker_bp.route('/api/reset', methods=['POST'])
def reset_tracker():
    global tracker_state, last_activity
    with _tracker_lock:
        tracker_state = {}
        last_activity = 0
    return jsonify({'ok': True})


def _ytdlp_bin():
    """Cari yt-dlp binary di venv atau PATH."""
    candidate = os.path.join(
        os.path.dirname(sys.executable),
        'yt-dlp.exe' if sys.platform == 'win32' else 'yt-dlp'
    )
    return candidate if os.path.exists(candidate) else 'yt-dlp'


def _ytdlp_run(url, extra_args=None, timeout=30):
    """Jalankan yt-dlp -g dengan argumen tambahan opsional."""
    fmt = 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best'
    cmd = [_ytdlp_bin(), '-g', '--no-playlist', '-f', fmt]
    if extra_args:
        cmd += extra_args
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _extract_stream_url(result):
    """Ambil URL pertama dari stdout yt-dlp."""
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    return lines[0] if lines else None


@object_tracker_bp.route('/api/cookies-status')
def cookies_status():
    exists = os.path.exists(COOKIES_FILE)
    size   = os.path.getsize(COOKIES_FILE) if exists else 0
    return jsonify({'exists': exists, 'size_kb': round(size / 1024, 1)})


@object_tracker_bp.route('/api/upload-cookies', methods=['POST'])
def upload_cookies():
    if 'cookies' not in request.files:
        return jsonify({'error': 'Tidak ada file cookies'}), 400
    f = request.files['cookies']
    if not f.filename:
        return jsonify({'error': 'Nama file kosong'}), 400
    os.makedirs(CONFIG_DIR, exist_ok=True)
    f.save(COOKIES_FILE)
    size = os.path.getsize(COOKIES_FILE)
    logger.info("Cookies file uploaded: %d bytes", size)
    return jsonify({'ok': True, 'size_kb': round(size / 1024, 1)})


@object_tracker_bp.route('/api/delete-cookies', methods=['POST'])
def delete_cookies():
    if os.path.exists(COOKIES_FILE):
        os.remove(COOKIES_FILE)
    return jsonify({'ok': True})


@object_tracker_bp.route('/api/resolve-url', methods=['POST'])
def resolve_url():
    """Resolve YouTube / URL stream dengan multi-strategy fallback."""
    data = request.get_json(silent=True) or {}
    url  = data.get('url', '').strip()

    if not url:
        return jsonify({'error': 'URL tidak boleh kosong'}), 400

    # ── Direct / HLS URL — langsung kirim ke frontend ─────────────────────
    url_lower = url.lower()
    if any(url_lower.endswith(ext) for ext in ('.m3u8', '.mp4', '.webm', '.ogg', '.mkv')):
        vtype = 'hls' if '.m3u8' in url_lower else 'direct'
        return jsonify({'url': url, 'type': vtype, 'source': 'direct'})

    # ── yt-dlp dengan berbagai strategi fallback ───────────────────────────

    strategies = [
        # 1. iOS player client — sering bypass bot-check tanpa login
        (['--extractor-args', 'youtube:player_client=ios'],
         'iOS client'),

        # 2. TV client — alternatif kedua
        (['--extractor-args', 'youtube:player_client=tv_embedded'],
         'TV client'),

        # 3. Cookies file (jika user sudah upload)
        (['--cookies', COOKIES_FILE] if os.path.exists(COOKIES_FILE) else None,
         'cookies file'),

        # 4. Browser cookies — coba Chrome, Firefox, Edge berurutan
        *[(['--cookies-from-browser', b], f'browser:{b}')
          for b in ('chrome', 'firefox', 'edge', 'chromium', 'brave')],

        # 5. Tanpa argumen tambahan (fallback terakhir)
        ([], 'default'),
    ]

    last_error = 'Tidak ada strategi yang berhasil.'

    for args, label in strategies:
        if args is None:
            continue   # strategi tidak applicable (e.g. cookies belum diupload)
        try:
            r = _ytdlp_run(url, args, timeout=35)
            if r.returncode == 0:
                stream_url = _extract_stream_url(r)
                if stream_url:
                    logger.info("resolve_url OK via %s: %s", label, url[:60])
                    return jsonify({
                        'url':    stream_url,
                        'type':   'direct',
                        'source': f'yt-dlp ({label})',
                    })
            else:
                last_error = (r.stderr.strip().splitlines() or [''])[-1]
                logger.debug("Strategy '%s' failed: %s", label, last_error[:120])
        except subprocess.TimeoutExpired:
            last_error = f'Timeout pada strategi {label}'
        except FileNotFoundError:
            return jsonify({'error': 'yt-dlp tidak ditemukan'}), 500

    # Semua strategi gagal → kembalikan pesan yang informatif
    need_cookies = 'Sign in' in last_error or 'bot' in last_error.lower()
    return jsonify({
        'error':       f'yt-dlp gagal: {last_error}',
        'need_cookies': need_cookies,
    }), 422


@object_tracker_bp.route('/api/track', methods=['POST'])
def track_frame():
    global tracker_state, last_activity

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image'}), 400

    raw   = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    lines_raw  = data.get('lines', [])
    distance_m = float(data.get('distance_m', 5.0))
    line_a = lines_raw[0] if len(lines_raw) > 0 else None
    line_b = lines_raw[1] if len(lines_raw) > 1 else None

    try:
        model = get_model()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    results = model.track(source=frame, persist=True, imgsz=320, conf=0.25, verbose=False)

    boxes_out  = []
    new_speeds = {}
    now = time.time()

    with _tracker_lock:
        if now - last_activity > TIMEOUT_SECONDS:
            tracker_state = {}
        last_activity = now

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            boxes   = result.boxes.data.cpu().numpy()
            ids_raw = result.boxes.id
            ids     = ids_raw.cpu().numpy() if ids_raw is not None else np.arange(len(boxes))
            classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.zeros(len(boxes))

            for box, raw_id, cls in zip(boxes, ids, classes):
                cls_int = int(cls)
                if cls_int not in ALLOWED_CLASSES:
                    continue

                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                conf   = float(box[4]) if len(box) > 4 else 1.0
                obj_id = str(int(raw_id))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if obj_id not in tracker_state:
                    tracker_state[obj_id] = {
                        'side_a': None, 'time_a': None,
                        'side_b': None, 'time_b': None,
                        'speed': 0.0,
                    }

                st = tracker_state[obj_id]

                # Crossing line A
                if line_a:
                    cur = _sign(_cross(cx, cy, line_a))
                    if cur != 0:
                        if st['side_a'] is None:
                            st['side_a'] = cur
                        elif st['side_a'] != cur:
                            st['side_a'] = cur
                            st['time_a'] = now
                            st['time_b'] = None   # reset agar urutan A→B benar

                # Crossing line B
                if line_b:
                    cur = _sign(_cross(cx, cy, line_b))
                    if cur != 0:
                        if st['side_b'] is None:
                            st['side_b'] = cur
                        elif st['side_b'] != cur:
                            st['side_b'] = cur
                            st['time_b'] = now

                # Kecepatan dihitung setelah melewati keduanya
                if st['time_a'] is not None and st['time_b'] is not None:
                    dt = abs(st['time_b'] - st['time_a'])
                    if dt > 0:
                        st['speed'] = distance_m / dt
                        new_speeds[obj_id] = st['speed']
                    st['time_a'] = None
                    st['time_b'] = None

                boxes_out.append({
                    'x': x1, 'y': y1, 'w': x2 - x1, 'h': y2 - y1,
                    'conf': conf,
                    'label': CLASS_NAMES.get(cls_int, f'cls{cls_int}'),
                    'id': obj_id,
                    'speed': st.get('speed', 0.0),
                })

        all_speeds = {oid: st['speed'] for oid, st in tracker_state.items() if st['speed'] > 0}

    return jsonify({
        'boxes': boxes_out,
        'new_speeds': {k: float(v) for k, v in new_speeds.items()},
        'all_speeds': {k: float(v) for k, v in all_speeds.items()},
    })


@object_tracker_bp.route('/api/upload-video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400
    upload_folder = current_app.config.get('UPLOAD_FOLDER', 'static/uploads')
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(file.filename)
    file.save(os.path.join(upload_folder, filename))
    return jsonify({'filename': filename})
