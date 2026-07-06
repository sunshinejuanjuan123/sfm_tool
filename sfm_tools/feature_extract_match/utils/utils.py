import os

env_value = os.getenv("SFM_IMG_PAIR_TIME_INTERVAL", "").strip()

try:
    SFM_IMG_PAIR_TIME_INTERVAL = float(env_value) if env_value else 0.5
except ValueError:
    print(f"Warning: Invalid SFM_IMG_PAIR_TIME_INTERVAL='{env_value}', using default value 2.")
    SFM_IMG_PAIR_TIME_INTERVAL = 0.5

print(f"SfM: SFM_IMG_PAIR_TIME_INTERVAL={SFM_IMG_PAIR_TIME_INTERVAL}")


def get_img_pairs(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(len(img_fnames)):
            if i == j:
                continue
            time0, _ = os.path.splitext(img_fnames[i].split("/")[-1])
            time1, _ = os.path.splitext(img_fnames[j].split("/")[-1])
            time0, time1 = float(time0)/1000, float(time1)/1000
            if abs(time0-time1) <= SFM_IMG_PAIR_TIME_INTERVAL:
                index_pairs.append((i, j))
    return index_pairs


def remove_db(database_path):
    if os.path.isfile(database_path):
        os.remove(database_path)
    if os.path.isfile(database_path+'-shm'):
        os.remove(database_path+'-shm')
    if os.path.isfile(database_path+'-wal'):
        os.remove(database_path+'-wal')
