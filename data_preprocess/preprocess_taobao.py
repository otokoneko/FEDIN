import os.path
import pickle as pkl
import random

import tqdm

from utils import neg_sample, truncate_list, delete_if_exists

PATH = r'./raw_data/taobao/'

START_TIME = 1511539200
SECONDS_PER_DAY = 24 * 3600
TEST_THRESHOLD = 1512230400

MAX_LENGTH = 100
TRAIN_NUM = 16
TEST_NUM = 4

HEADER = 'clk,uid,iid,cid,iid_his,cid_his'


def feateng(in_file, remap_dict_file):
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}

    uid_set = set()
    iid_set = set()
    cid_set = set()

    with open(in_file, 'r') as r:
        for line in r:
            uid, iid, cid, btype, ts = line.split(',')
            if btype == 'pv':
                uid_set.add(uid)
                iid_set.add(iid)
                cid_set.add(cid)

    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)

    feature_id = 1
    for uid in uid_list:
        uid_remap_dict[uid] = str(feature_id)
        feature_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = str(feature_id)
        feature_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = str(feature_id)
        feature_id += 1
    print('total original feature number: {}'.format(feature_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
    print('remap dict dumpped')


def remap_log_file(input_log_file, remap_dict_file, output_log_file):
    with open(remap_dict_file, 'rb') as f:
        uid_remap_dict = pkl.load(f)
        iid_remap_dict = pkl.load(f)
        cid_remap_dict = pkl.load(f)
    user_seq_dict = {}
    item_feat_dict = {}
    newlines = []

    with open(input_log_file, 'r') as f:
        for line in f:
            uid, iid, cid, btype, ts = line[:-1].split(',')
            if btype != 'pv':
                continue
            uid = uid_remap_dict[uid]
            iid = iid_remap_dict[iid]
            cid = cid_remap_dict[cid]

            if uid not in user_seq_dict:
                user_seq_dict[uid] = [iid]
            else:
                user_seq_dict[uid].append(iid)

            if iid not in item_feat_dict:
                item_feat_dict[iid] = [cid]

            newline = ','.join([uid, iid, cid, ts]) + '\n'
            newlines.append(newline)

    with open(output_log_file, 'w') as f:
        f.writelines(newlines)

    print('remap log file dumpped')
    delete_if_exists(remap_dict_file)


def gen_samples(user_seq_dict, item_set, uid):
    train_list = []
    test_list = []
    user_seq = user_seq_dict[uid]
    for idx in range(len(user_seq)):
        iid, cid, ts = user_seq[idx]
        item = (iid, cid)
        clk = random.randint(0, 1)
        if clk == 0:
            item = neg_sample(item_set, [x[:-1] for x in user_seq[:idx]], item)
        history = user_seq[:idx][-MAX_LENGTH:]
        iid_his = '^'.join(x[0] for x in history)
        cid_his = '^'.join(x[1] for x in history)
        sample = ','.join([str(clk), uid, item[0], item[1], iid_his, cid_his]) + '\n'
        if ts < TEST_THRESHOLD:
            train_list.append(sample)
        else:
            test_list.append(sample)
    return train_list, test_list


def gen_target_seq(input_file, target_train_file, target_vali_file, target_test_file):
    user_seq_dict = {}
    item_set = set()

    with open(input_file, 'r') as f:
        for line in f:
            uid, iid, cid, ts = line[:-1].split(',')
            ts = int(ts)
            item_set.add((iid, cid))
            if uid not in user_seq_dict:
                user_seq_dict[uid] = [(iid, cid, ts)]
            else:
                user_seq_dict[uid].append((iid, cid, ts))

    item_set = list(item_set)

    f_train = open(target_train_file, 'w')
    f_vali = open(target_vali_file, 'w')
    f_test = open(target_test_file, 'w')

    f_train.write(f'{HEADER}\n')
    f_vali.write(f'{HEADER}\n')
    f_test.write(f'{HEADER}\n')

    for uid in tqdm.tqdm(user_seq_dict.keys()):
        if len(user_seq_dict[uid]) > 3:
            train_list, test_list = gen_samples(user_seq_dict, item_set, uid)
            train_list = truncate_list(train_list, TRAIN_NUM)
            test_list = truncate_list(test_list, TEST_NUM)
            mid = len(test_list) // 2
            vali_list = test_list[:mid]
            test_list = test_list[mid:]
            f_train.writelines(train_list)
            f_vali.writelines(vali_list)
            f_test.writelines(test_list)

    f_train.close()
    f_vali.close()
    f_test.close()


def main():
    raw_path = os.path.join(PATH, 'UserBehavior.csv')
    id_remap_dict = os.path.join(PATH, 'id_remap_dict.pkl')
    remapped_log = os.path.join(PATH, 'remapped_log.csv')
    train_path = os.path.join(PATH, 'train.csv')
    valid_path = os.path.join(PATH, 'valid.csv')
    test_path = os.path.join(PATH, 'test.csv')

    if os.path.exists(remapped_log):
        gen_target_seq(remapped_log, train_path, valid_path, test_path)
        return

    feateng(raw_path, id_remap_dict)
    remap_log_file(raw_path, id_remap_dict, remapped_log)
    gen_target_seq(remapped_log, train_path, valid_path, test_path)


if __name__ == "__main__":
    main()

