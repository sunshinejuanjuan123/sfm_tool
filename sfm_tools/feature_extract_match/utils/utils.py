import os

def get_img_pairs(img_fnames):
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(len(img_fnames)):
            if i == j:
                continue
            time0, _ = os.path.splitext(img_fnames[i].split("/")[-1])
            time1, _ = os.path.splitext(img_fnames[j].split("/")[-1])
            if abs(float(time0) - float(time1)) <= 5.0:
                index_pairs.append((i, j))
    return index_pairs

def remove_db(database_path):
    if os.path.isfile(database_path):
        os.remove(database_path)
    if os.path.isfile(database_path+'-shm'):
        os.remove(database_path+'-shm')
    if os.path.isfile(database_path+'-wal'):
        os.remove(database_path+'-wal')