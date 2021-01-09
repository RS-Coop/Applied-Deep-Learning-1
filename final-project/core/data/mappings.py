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
    1: 'pedestrian',
    2: 'road',
    3: 'sidewalk',
    4: 'moveable-object',
    5: 'terrain',
    6: 'building',
    7: 'car',
    8: 'truck',
    9: 'bicycle',
    10: 'motorcycle',
    11: 'bus',
    12: 'vegetation',
    13: 'ground',
    14: 'animal',
    15: 'static-object',
    16: 'vehicle-other'
}

#Mapping from nuscnes label to model label
#nuscenes labels to my subset
nuscenes2model = {
    0: 0,
    1: 14, 
    2: 1, 
    3: 1, 
    4: 1, 
    5: 1, 
    6: 1, 
    7: 1, 
    8: 1, 
    9: 4, 
    10: 4, 
    11: 4, 
    12: 4, 
    13: 15, 
    14: 9, 
    15: 11, 
    16: 11, 
    17: 7, 
    18: 16, 
    19: 7, 
    20: 7, 
    21: 10, 
    22: 16, 
    23: 8, 
    24: 2, 
    25: 13, 
    26: 3, 
    27: 5, 
    28: 6, 
    29: 15, 
    30: 12, 
    31: 0
}

#####################################################################################
'''
Semantic KITTI mappings
'''
#Mapping from semantic KITTI labels to name
kitti_label2name = {
    0 : "unlabeled",
    1 : "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle"
}

#Mapping from model label to name
#Only keeping subset of all KITTI labels
#This is the same as SPVNAS
model_label2name= {
    1: 'car', 2: 'bicycle', 3: 'motorcycle', 4: 'truck', 5:
    'other-vehicle', 6: 'person', 7: 'bicyclist', 8: 'motorcyclist',
    9: 'road', 10: 'parking', 11: 'sidewalk', 12: 'other-ground',
    13: 'building', 14: 'fence', 15: 'vegetation', 16: 'trunk',
    17: 'terrain', 18: 'pole', 19: 'traffic-sign'
}

#Mapping from semantic KITTI labels to kept model labels
kitti2model = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "car"
    253: 2,    # "bicyclist"
    254: 6,    # "person"
    255: 8,    # "motorcyclist"
    256: 5,    # "other-vehicle" ------mapped
    257: 5,    # "other-vehicle" -----------mapped
    258: 4,    # "truck"
    259: 5    # "other-vehicle"
}
