import copy

def mark_active_menu(path, menu_items):
    path = path.rstrip('/')  # hapus trailing slash supaya konsisten
    menu_items = copy.deepcopy(menu_items)
    for menu in menu_items:
        menu['active'] = False
        if 'children' in menu and menu['children']:
            for child in menu['children']:
                child_url = child['url'].rstrip('/')
                child['active'] = (child_url == path)
                if child['active']:
                    menu['active'] = True
        else:
            menu['active'] = (menu['url'].rstrip('/') == path)
    return menu_items
