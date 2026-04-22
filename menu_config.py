menus = [
    {
        "name": "Home",
        "url": "/",
        "icon": "bi-house"
    },
    {
        "name": "Object-Detection",
        "url": "#",
        "icon": "bi-camera",
        "children": 
        [
            {"name": "Real-Time", "url": "/object-detection/camera"},
            {"name": "Image-Segmentation", "url": "/segmentation/camera"},
        ],
    },
    {
        "name": "Tools",
        "url": "#",
        "icon": "bi bi-tools",
        "children": [
            {"name": "Upload-Video", "url": "/object-detection/video/upload"},
            {"name": "Upload-File", "url": "/object-detection/file-multi"},
            {"name": "SEG-Upload-Video", "url": "/segmentation/video"},
            {"name": "SEG-Upload-File", "url": "/segmentation/image"},
        ],
    },
    {
        "name": "Experimental",
        "url": "#",
        "icon": "bi-image",
        "children": [
            {"name": "Mask R-CNN", "url": "/under-construction"},
            {"name": "Object-tracker", "url": "/object-tracker/camera"},
            {"name": "Segmentation Image", "url": "/object-tracker-segmentation"},
            {"name": "Mask R-CNN++", "url": "/under-construction","icon": "bi-bounding-box-circles"},
        ],
    },
    {"name": "Auto Annotation", "url": "/experimental/auto-annotation", "icon": "bi-pencil-square"},
    {
        "name": "Mask R-CNN",
        "url": "#",
        "icon": "bi-bounding-box-circles"
    },
    {
    "name": "Experimental SAHI Segmentasi",
    "url": "/experimental/auto-annotation-sahi",
    "icon": "bi-brush",
},
    {
        'name': 'DATA',
        'url': '#',
        'icon': 'bi-folder',
        'children': [
            # menu lain...
            {'name': 'Edit Data', 'url': '/data/edit'}
        ],
        'active': False
    },
    {
        "name": "Setting",
        "url": "#",
        "icon": "bi-gear",
        "children": [
            {"name": "Profile", "url": "/under-construction"},
            {"name": "Account", "url": "/under-construction"},
            {"name": "Camera", "url": "/setting/camera"},
        ],
    },
]
