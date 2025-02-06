import os, warnings
from tqdm import tqdm
import h5py
from PIL import Image, ExifTags

# 3D reconstruction
# import pycolmap
import sys
import sqlite3
import numpy as np
import time
from read_write_model import read_cameras_text

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)


    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(self, model, width, height, params,
                   prior_focal_length=False, camera_id=None):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (camera_id, model, width, height, array_to_blob(params),
             prior_focal_length))
        return cursor.lastrowid

    def add_image(self, name, camera_id,
                  prior_q=np.zeros(4), prior_t=np.zeros(3), image_id=None):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, name, camera_id, prior_q[0], prior_q[1], prior_q[2],
             prior_q[3], prior_t[0], prior_t[1], prior_t[2]))
        return cursor.lastrowid

    def add_keypoints(self, image_id, keypoints):
        assert(len(keypoints.shape) == 2)
        assert(keypoints.shape[1] in [2, 4, 6])

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),))

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),))

    def add_matches(self, image_id1, image_id2, matches):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),))

    def add_two_view_geometry(self, image_id1, image_id2, matches,
                              F=np.eye(3), E=np.eye(3), H=np.eye(3), config=2):
        assert(len(matches.shape) == 2)
        assert(matches.shape[1] == 2)

        if image_id1 > image_id2:
            matches = matches[:,::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches), config,
             array_to_blob(F), array_to_blob(E), array_to_blob(H)))


def create_camera(db, model, width, height, param_arr):
    return db.add_camera(model, width, height, param_arr)


def add_keypoints(db, h5_path, image_path, img_ext, sparse_init_dir):
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    camera_id = None
    fname_to_id = {}

    cameras_init = read_cameras_text(os.path.join(sparse_init_dir, "cameras.txt"))
    
    # model id 1
    for camera_id in cameras_init.keys():
        _ = create_camera(db, 1, width=cameras_init[camera_id].width, height=cameras_init[camera_id].height, param_arr=cameras_init[camera_id].params)
    
    camera_name_2_id = {'center_camera_fov120': 1,
                        'left_front_camera': 2,
                        'left_rear_camera': 3,
                        'right_front_camera': 4,
                        'right_rear_camera': 5,
                        'rear_camera': 6,
                        'center_camera_fov30': 7}

    for camera_name in tqdm(list(keypoint_f.keys()), desc='add_keypoints'):
        for image_name in tqdm(list(keypoint_f[camera_name].keys())):
            keypoints = keypoint_f[camera_name][image_name][()]
            fname_with_ext = camera_name + "/" + image_name
            path = os.path.join(image_path, fname_with_ext)
            if not os.path.isfile(path):
                raise IOError(f'Invalid image path {path}')
            image_id = db.add_image(fname_with_ext, camera_name_2_id[camera_name])
            fname_to_id[fname_with_ext] = image_id
            db.add_keypoints(image_id, keypoints)

    return fname_to_id

def check_name(fname1, fname2):
    
    fname1_class = int(fname1[:-4].split("_")[1])
    fname2_class = int(fname2[:-4].split("_")[1])

    if fname1_class == 0:
        return False
    elif fname1_class == 1:
        if fname2_class in [0]:
            return False
        return True
    else:
        if fname2_class in [0]:
            return False
        return True

def add_matches(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, 'matches_adalam.h5'), 'r')

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    for camera_key_1 in tqdm(match_file.keys(), desc='add_matches'):
        for img_name_1 in tqdm(match_file[camera_key_1].keys()):
            for camera_key_2 in tqdm(match_file[camera_key_1][img_name_1].keys()):
                    group = match_file[camera_key_1][img_name_1][camera_key_2]
                    for img_name_2 in group.keys():
                        key_1 = camera_key_1 + "/" + img_name_1
                        key_2 = camera_key_2 + "/" + img_name_2
                        id_1 = fname_to_id[key_1]
                        id_2 = fname_to_id[key_2]
                        pair_id = image_ids_to_pair_id(id_1, id_2)
                        if pair_id in added:
                            print(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                            continue
                        matches = group[img_name_2][()]
                        db.add_matches(id_1, id_2, matches)
                        added.add(pair_id)


def add_two_view_geometry(db, h5_path, fname_to_id):
    match_file = h5py.File(os.path.join(h5_path, 'matches_adalam.h5'), 'r')

    added = set()
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    for camera_key_1 in tqdm(match_file.keys(), desc='add_two_view_geometry'):
        for img_name_1 in tqdm(match_file[camera_key_1].keys()):
            for camera_key_2 in tqdm(match_file[camera_key_1][img_name_1].keys()):
                group = match_file[camera_key_1][img_name_1][camera_key_2]
                for img_name_2 in group.keys():
                    key_1 = camera_key_1 + "/" + img_name_1
                    key_2 = camera_key_2 + "/" + img_name_2
                    id_1 = fname_to_id[key_1]
                    id_2 = fname_to_id[key_2]

                    pair_id = image_ids_to_pair_id(id_1, id_2)
                    if pair_id in added:
                        print(f'Pair {pair_id} ({id_1}, {id_2}) already added!')
                        continue
                    matches = group[img_name_2][()]
                    db.add_two_view_geometry(id_1, id_2, matches)
                    added.add(pair_id)


def import_into_colmap(img_dir,
                       feature_dir='.featureout_copy',
                       database_path='colmap.db',
                       img_ext='.jpg',
                       sparse_init_dir=None):
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, sparse_init_dir)
    add_matches(db, feature_dir, fname_to_id)
    add_two_view_geometry(db, feature_dir, fname_to_id)
    db.commit()
    db.close()
    return
