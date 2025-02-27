import numpy as np
from scipy import sparse
import numpy as np
import os
from tqdm import tqdm
import math 
import random
import base64
import cv2
from datetime import datetime
import time
import threading
import sys
sys.path.append(os.path.abspath('..'))
import config.path, config.system
# from base.minio_connector import MinioConnector
# from base.db import MongoDB,BaseCassanda,DBInfor_test
# import src.similarity as similarity
import faiss
import pickle

# mongodb = MongoDB()
# cassandra_connection_29 = BaseCassanda(cassandra_hosts=[config.system.CASS_HOST],cassandra_keyspace='facesearch')
# fb_db_test = DBInfor_test(cassandra_connection_29)
# minio_connector = MinioConnector()

def check_train(id):
    infos = fb_db_test.get_info(id)
    return infos['train']

# def train_update (number_cluster, name_index, ids, embs, save_path) :
    
#     print("Trainning IVFFLATL2 {} start".format(number_cluster))
    
#     index = similarity.IndexIVFFLATL2(n_cluster = number_cluster)

#     if number_cluster == 65536:
#         # print(os.path.join(save_path, name_index+'_10M_ids.pickle'))
#         print("Trainning IVFFLATL2 {}".format(number_cluster)) 
#         current_path_emb = os.path.join(name_index,'embs/')
#         ids_emb, embs_np = load_embs(current_path_emb)

#         with open( os.path.join(save_path, name_index+'_10M_ids.pickle'), 'wb') as ids_file:
#             pickle.dump(ids_emb, ids_file)
        
#         index.train(embs_np)
#         index.add(embs_np)
#         faiss.write_index(index, os.path.join(save_path ,(name_index+'_10M.index')))

#     elif number_cluster == 16384 :
#         print("Trainning IVFFLATL2 {}".format(number_cluster)) 
#         index.train(embs)
#         faiss.write_index(index, os.path.join(save_path ,(name_index+'.index')))
        
#         for id in ids:
#             fb_db_test.update_info_train(train_value='True', id = id)
#     else:
#         print("Wrong number of cluster")
#         pass
#     print("Trainning IVFFLATL2 {} done".format(number_cluster))
#     # ### update_train_status ###
   

def check_exist(id):
    
    infos = fb_db_test.get_info(id)
    if infos is not None:
        return True
    else:
        return False

def save_fb(fbid_path, fb_db, emb_path):
    # print("savefb")
    label_path = os.path.join(fbid_path, 'labels.txt')
    fbid = fbid_path.split('/')[-1]

    if os.path.isfile(label_path):
        with open(label_path, 'r') as f:
            data = f.read()
        
        data = data.split('\n')

        # records = []
        name = []
        label = []
        for line in data:
            name.append(line.split('\t')[0])
            label.append(line.split('\t')[-1])
        name.remove('')
        label.remove('')
        
        name = np.asarray(name)
        label = np.asarray(label)
        for i in range(name.shape[0]):
            id = fbid + '_' + name[i]
            path = os.path.join(fbid_path, (name[i] + '.jpeg'))
            e_path = os.path.join(emb_path, fbid)
            e_path = os.path.join(e_path, (name[i] + '.npy'))
            # emb = np.load(e_path)
            # emb_str = emb.tostring()

            idx_same = np.where(label==label[i])
            name_same = name[idx_same]
            name_same = list(name_same)
            name_same = [os.path.join(fbid_path, (f + '.jpeg')) for f in name_same]
            name_same = [f for f in name_same if f != path]
            name_same = [str(f) for f in name_same]

            idx_diff = np.where(label!=label[i])
            name_diff = name[idx_diff]
            name_diff = list(name_diff)
            name_diff = [os.path.join(fbid_path, (f + '.jpeg')) for f in name_diff]
            name_diff = [str(f) for f in name_diff]
            label_diff = label[idx_diff]
            label_diff = list(label_diff)
            label_diff = [str(f) for f in label_diff]
            
            record = {'id': id, 'path': path, 'same': name_same, 'diff': name_diff, 'label_diff': label_diff, 'emb_path': e_path}

            fb_db.set_fbimage(record)
            #print(record)

def save_info2db (list_names, labels, fbid, img_save_path, emb_save_path,source_info='facebook',train='False'):   
    name_np = np.asarray(list_names)
    label_np = np.asarray(labels)
    
    for i in range(name_np.shape[0]):
        id = fbid + '_' + name_np[i]

        e_path = emb_save_path + '_' + name_np[i] + '.npy'
        i_path = img_save_path + '_' + name_np[i] + '.jpeg'
        # emb = np.load(e_path)
        # emb_str = emb.tostring()

        idx_same = np.where(label_np==label_np[i])

        name_same = name_np[idx_same]

        name_same = list(name_same)
        
        name_same = [img_save_path + '_' + f + '.jpeg' for f in name_same]

        name_same = [f for f in name_same if f != i_path]
        
        name_same = [str(f) for f in name_same]

        idx_diff = np.where(label_np!=label_np[i])
        
        name_diff = name_np[idx_diff]
        name_diff = list(name_diff)
        name_diff = [img_save_path + '_' + f + '.jpeg' for f in name_diff]

        name_diff = [str(f) for f in name_diff]
        
        label_diff = label_np[idx_diff]
        label_diff = list(label_diff)
        label_diff = [str(f) for f in label_diff]

        if source_info == 'facebook':
            source_link_full = 'https://facebook.com/' + name_np[i].split("#")[0]
            source = source_info
        elif source_info == 'ekyc':
            source_link_full = ''
            source = source_info

        elif source_info == 'newstinh':
    
            source_news = fbid.split('_')[0]
            id_news = fbid.split('_')[1]
            source_link_full = mongodb.get_article(id = id_news,news_name= source_news[3:],db_name='news_db_63tinh')['url']### get data in mongoDB
            # print(source_link_full)
            source = source_news

        elif source_info == 'dantri' or source_info == 'vnexpress':
            
            id_news = fbid
            source_link_full = mongodb.get_article(id= id_news,news_name=source_info,db_name='news_db')['url'] ### get data in mongoDB
            source = source_info
        else:
            source_link_full = ''
            source = source_info

        record = {'id': id, 'img_path': i_path, 'emb_path': e_path, 'same': name_same, 'maybe_known': name_diff, 'maybe_known_label': label_diff, \
                    'source': source, 'source_link': source_link_full, 'train': train, 'createdat': int(time.time() * 1)}
        print(record)
        fb_db_test.set_info(record)

def check_face(emb_query, list_emb):
    sims = consine_distance(np.asarray(emb_query), np.asarray(list_emb))
    iu = np.triu_indices(sims.shape[1])
    sims = np.tril(sims, k=-1)
    sims[iu] = -1
    idxs = np.argmax(sims, axis=1)
    return sims, idxs

def square_crop(im, S):
    if im.shape[0] > im.shape[1]:
        height = S
        width = int(float(im.shape[1]) / im.shape[0] * S)
        scale = float(S) / im.shape[0]
    else:
        width = S
        height = int(float(im.shape[0]) / im.shape[1] * S)
        scale = float(S) / im.shape[1]
    resized_im = cv2.resize(im, (width, height))
    det_im = np.zeros((S, S, 3), dtype=np.uint8)
    det_im[:resized_im.shape[0], :resized_im.shape[1], :] = resized_im
    return det_im, scale

def consine_similarity(array1, array2):
    arr1 = array1
    arr2 = array2
    
    result = 0
    result = np.sum(np.multiply(arr1, arr2))
    size_arr1 = np.sum(np.multiply(arr1, arr1))
    size_arr2 = np.sum(np.multiply(arr2, arr2))
    try:
        result = result / math.sqrt(size_arr1 * size_arr2)
    except:
        result = 0
    return result

def consine_distance(arr1, arr2):
    if len(arr1.shape) == 1:
        arr1 = arr1[np.newaxis, :]
    if len(arr2.shape) == 1:
        arr2 = arr2[np.newaxis, :]

    multi = np.dot(arr1, arr2.T) 
    norm1 = np.linalg.norm(arr1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(arr2, axis=1, keepdims=True)
    norm_mul = np.dot(norm1, norm2.T)
    try:
        result = np.divide(multi, norm_mul)
    except:
        result = np.zeros(arr1.shape[0], arr2.shape[0])
    return result

def save_img(img, name):
    cv2.imwrite(name, img)

def base64_to_image(base64_string):
    base64_string = base64_string.replace('data:image/jpeg;base64,','')
    base64_string = base64_string.replace('data:image/jpg;base64,', '')
    base64_string = base64_string.replace('data:image/png;base64,', '')

    im_bytes = base64.b64decode(base64_string)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def save_log_image(img, root_path, folder_name, prefix='face'):
    today = datetime.now()

    log_folder = os.path.join(root_path, folder_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    filename = str(int(time.time()*1000.0)) + '.jpg'
    img_path = os.path.join(log_folder, filename)
    th = threading.Thread(target=save_img, args=(img[:,:,::-1], img_path))
    th.start()
    #cv2.imwrite(log_folder +'/'+ filename, img)
    return filename

def save_log_emb(emb, root_path, folder_name, filename, prefix='face'):
    today = datetime.now()

    log_folder = os.path.join(root_path, folder_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    emb_name = filename.split('.')[0]
    emb_path = os.path.join(log_folder, emb_name)
    np.save(emb_path, emb)
    return filename

def save_predicted_image(img, filename):
    today = datetime.now()

    predicted_folder = config.path.PREDICTED_FOLDER + today.strftime('%Y%m%d')
    if not os.path.exists(predicted_folder):
        os.makedirs(predicted_folder)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_path = predicted_folder +'/'+ filename
    th = threading.Thread(target=save_img, args=(img, img_path))
    th.start()
    # cv2.imwrite(predicted_folder +'/'+ filename, img)

    return predicted_folder +'/'+ filename

# Caculator consine similarity
def consine_similarity(array1, array2):
    arr1 = array1
    arr2 = array2
    
    result = 0
    result = np.sum(np.multiply(arr1, arr2))
    size_arr1 = np.sum(np.multiply(arr1, arr1))
    size_arr2 = np.sum(np.multiply(arr2, arr2))
    try:
        result = result / math.sqrt(size_arr1 * size_arr2)
    except:
        result = 0
    return result

# data cau truc image->list image or image->person->list image
def stack_embs(path, lv2=False):
    if not lv2:
        emb_list = os.listdir(path)
        emb_list = sorted(emb_list)

        xb = []
        ids =[]
        for p in tqdm(emb_list):
            id = p.split('.')[0]
            emb_np = np.load(os.path.join(path, p))
            ids.append(id)
            xb.append(emb_np)

        #ids_np = np.stack(ids)
        xb_np = np.stack(xb)
        # np.save('emb_1m', xb_np)
    else:
        cnt = 0
        folders = [os.path.join(path, f) for f in os.listdir(path)]
        folders = sorted(folders)
        #folders = folders[1112253:]
        xb = []
        ids =[]

        for folder in tqdm(folders):
            try:
                emb_list = [os.path.join(folder, f) for f in os.listdir(folder)]
                emb_list = sorted(emb_list)

                for p in emb_list:
                    id = p.split('/')[-2] + '_' + p.split('/')[-1].split('.')[0]
                    emb_np = np.load(p)
                    ids.append(id)
                    xb.append(emb_np)
                    cnt += 1
            except:
                continue

        #ids_np = np.stack(ids)
        xb_np = np.stack(xb)
        # np.save("emb_1m", xb_np)
    return ids, xb_np

def load_embs(path):
    
    folders = minio_connector.list_objects(path)
    # folders = sorted(folders)
    xb = []
    ids =[]        
    for folder in tqdm(folders):
        id = folder.split('/')[-1].split(".")[0]
        # print(id)
        # emb_path = os.path.join(folder, id + '_per.npy')
        emb_np = minio_connector.download_emb(folder)
        ids.append(id)
        xb.append(emb_np)

    if len(xb) > 0:
        xb_np = np.stack(xb)
    else:
        xb_np = np.array([])
    return ids, xb_np
