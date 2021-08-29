def load_activity_map_wHAR():
    map = {}
    map[0] = 'jump'
    map[1] = 'lie down'
    map[2] = 'sit'
    map[3] = 'stand'
    map[4] = 'walk'
    map[5] = 'stairs up'
    map[6] = 'stairs down'
    map[7] = 'transition'
    return map

def load_activity_map_har():
    map = {}
    map[0] = 'walking'
    map[1] = 'walking_upstairs'
    map[2] = 'walking_downstairs'
    map[3] = 'sitting'
    map[4] = 'standing'
    map[5] = 'laying'
    return map

def load_activity_map_mhealth():
    map = {}
    map[0] = 'standing'
    map[1] = 'sitting'
    map[2] = 'lying_down'
    map[3] = 'walking'
    map[4] = 'climbing_stairs'
    map[5] = 'waist_bends'
    map[6] = 'frontal_elevation'
    map[7] = 'crouching'
    map[8] = 'cycling'
    map[9] = 'jogging'
    map[10] = 'running'
    map[11] = 'jumping'
    return map

def load_activity_map_opportunity():
    map = {}
    map[0] = 'open_door_1'
    map[1] = 'open_door_2'
    map[2] = 'close_door_1'
    map[3] = 'close_door_2'
    map[4] = 'open_fridge'
    map[5] = 'close_fridge'
    map[6] = 'open_dishwasher'
    map[7] = 'close_dishwasher'
    map[8] = 'open_drawer_1'
    map[9] = 'close_drawer_1'
    map[10] = 'open_drawer_2'
    map[11] = 'close_drawer_2'
    map[12] = 'open_drawer_3'
    map[13] = 'close_drawer_3'
    map[14] = 'clean_table'
    map[15] = 'drink'
    map[16] = 'toggle_switch'
    return map