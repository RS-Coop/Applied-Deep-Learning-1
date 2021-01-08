'''
NuScenes mappings
'''
#Mapping from nuscnes label to nuscenes name
nuscenes_label2name = {
    0: 'noise', 
    1: 'animal', 
    2: 'human.pedestrian.adult', 
    3: 'human.pedestrian.child', 
    4: 'human.pedestrian.construction_worker', 
    5: 'human.pedestrian.personal_mobility', 
    6: 'human.pedestrian.police_officer', 
    7: 'human.pedestrian.stroller', 
    8: 'human.pedestrian.wheelchair', 
    9: 'movable_object.barrier', 
    10: 'movable_object.debris', 
    11: 'movable_object.pushable_pullable', 
    12: 'movable_object.trafficcone', 
    13: 'static_object.bicycle_rack', 
    14: 'vehicle.bicycle', 
    15: 'vehicle.bus.bendy', 
    16: 'vehicle.bus.rigid', 
    17: 'vehicle.car', 
    18: 'vehicle.construction', 
    19: 'vehicle.emergency.ambulance', 
    20: 'vehicle.emergency.police', 
    21: 'vehicle.motorcycle', 
    22: 'vehicle.trailer', 
    23: 'vehicle.truck', 
    24: 'flat.driveable_surface', 
    25: 'flat.other', 
    26: 'flat.sidewalk', 
    27: 'flat.terrain', 
    28: 'static.manmade', 
    29: 'static.other', 
    30: 'static.vegetation', 
    31: 'vehicle.ego'
}

#Mapping from model label to name
#I only keep a subset of the nuscenes labels
model_label2name = {
    0: 'pedestrian',
    1: 'road',
    2: 'sidewalk',
    3: 'moveable-object',
    4: 'terrain',
    5: 'building',
    6: 'car',
    7: 'truck',
    8: 'bicycle',
    9: 'motorcycle',
    10: 'bus',
    11: 'vegetation',
    12: 'ground',
    13: 'noise',
    14: 'animal',
    15: 'static-object',
    16: 'vehicle-other'
}

#Mapping from nuscnes label to model label
#nuscenes labels to my subset
nuscenes2model = {
    0: 13,
    1: 14, 
    2: 0, 
    3: 0, 
    4: 0, 
    5: 0, 
    6: 0, 
    7: 0, 
    8: 0, 
    9: 3, 
    10: 3, 
    11: 3, 
    12: 3, 
    13: 15, 
    14: 8, 
    15: 10, 
    16: 10, 
    17: 6, 
    18: 16, 
    19: 6, 
    20: 6, 
    21: 9, 
    22: 16, 
    23: 7, 
    24: 1, 
    25: 12, 
    26: 2, 
    27: 4, 
    28: 5, 
    29: 5, 
    30: 11, 
    31: 6
}

#####################################################################################
'''
Semantic KITTI mappings
'''
#Mapping from semantic KITTI labels to kept model labels
kitti2model = {

}

#####################################################################################
'''
SPVNAS mappings
'''
#Mapping from SPVNAS output to name
#This is the mapping from the e3d SPVNAS Semantic KITTI
spvnas_label2name = {
        0: 'car', 1: 'bicycle', 2: 'motorcycle', 3: 'truck', 4:
        'other-vehicle', 5: 'person', 6: 'bicyclist', 7: 'motorcyclist',
        8: 'road', 9: 'parking', 10: 'sidewalk', 11: 'other-ground',
        12: 'building', 13: 'fence', 14: 'vegetation', 15: 'trunk',
        16: 'terrain', 17: 'pole', 18: 'traffic-sign'
    }

#Mapping from SPVNAS to model
#From e3d to my labels
spvnas2model = {
    0: 6,
    1: 8,
    2: 9,
    3: 7,
    4: 16,
    5: 0,
    6: 8,
    7: 9,
    8: 1,
    9: 1,
    10: 2,
    11: 12,
    12: 5,
    13: 15,
    14: 11,
    15: 11,
    16: 4,
    17: 15,
    18: 15,
}

#Mapping from model labels to SPVNAS labels
#
'''
Currently a few of the labels are mapped to 255 which is the ignore label.
This is because there is no clear cut map, but it would be ideal to have
multiple valid labels. Mostly this is just for some basic testing so its not too
big of a deal.
'''
model2spvnas = {
    0: 5,
    1: 8,
    2: 10,
    3: 255,
    4: 16,
    5: 12,
    6: 0,
    7: 3,
    8: 1,
    9: 2,
    10: 4,
    11: 14,
    12: 11,
    13: 255,
    14: 255,
    15: 255,
    16: 4
}