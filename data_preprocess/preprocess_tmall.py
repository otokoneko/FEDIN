import datetime
import os.path
import pickle as pkl
import random
import time

import tqdm

from utils import neg_sample, truncate_list, delete_if_exists

PATH = r'./raw_data/tmall/'

HEADER = 'clk,uid,aid,gid,iid,cid,sid,bid,iid_his,cid_his,sid_his,bid_his'
TEST_THRESHOLD = 1444838400

MAX_LENGTH = 100
TRAIN_NUM = 32
TEST_NUM = 8
REMOVE_DUP_THRESHOLD = 0


def join_user_profile(user_profile_file, behavior_file, joined_file):
    user_profile_dict = {}
    with open(user_profile_file, 'r') as f:
        for line in f:
            uid, aid, gid = line[:-1].split(',')
            user_profile_dict[uid] = ','.join([aid, gid])

    # join
    newlines = []
    with open(behavior_file, 'r') as f:
        for line in f:
            uid = line[:-1].split(',')[0]
            user_profile = user_profile_dict[uid]
            newlines.append(line[:-1] + ',' + user_profile + '\n')
    with open(joined_file, 'w') as f:
        f.writelines(newlines)


def feateng(joined_raw_file, remap_dict_file):
    uid_set = set()
    iid_set = set()
    cid_set = set()
    sid_set = set()
    bid_set = set()
    aid_set = set()
    gid_set = set()
    with open(joined_raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            uid, iid, cid, sid, bid, date_str, btypeid, aid, gid = line[:-1].split(',')
            uid_set.add(uid)
            iid_set.add(iid)
            cid_set.add(cid)
            sid_set.add(sid)
            bid_set.add(bid)
            aid_set.add(aid)
            gid_set.add(gid)
            date_str = '2015' + date_str
            time_int = int(time.mktime(datetime.datetime.strptime(date_str, "%Y%m%d").timetuple()))

    # remap
    uid_list = list(uid_set)
    iid_list = list(iid_set)
    cid_list = list(cid_set)
    sid_list = list(sid_set)
    bid_list = list(bid_set)
    aid_list = list(aid_set)
    gid_list = list(gid_set)

    print('user num: {}'.format(len(uid_list)))
    print('item num: {}'.format(len(iid_list)))
    print('cate num: {}'.format(len(cid_list)))
    print('seller num: {}'.format(len(sid_list)))
    print('brand num: {}'.format(len(bid_list)))
    print('age num: {}'.format(len(aid_list)))
    print('gender num: {}'.format(len(gid_list)))

    remap_id = 1
    uid_remap_dict = {}
    iid_remap_dict = {}
    cid_remap_dict = {}
    sid_remap_dict = {}
    bid_remap_dict = {}
    aid_remap_dict = {}
    gid_remap_dict = {}

    for uid in uid_list:
        uid_remap_dict[uid] = str(remap_id)
        remap_id += 1
    for iid in iid_list:
        iid_remap_dict[iid] = str(remap_id)
        remap_id += 1
    for cid in cid_list:
        cid_remap_dict[cid] = str(remap_id)
        remap_id += 1
    for sid in sid_list:
        sid_remap_dict[sid] = str(remap_id)
        remap_id += 1
    for bid in bid_list:
        bid_remap_dict[bid] = str(remap_id)
        remap_id += 1
    for aid in aid_list:
        aid_remap_dict[aid] = str(remap_id)
        remap_id += 1
    for gid in gid_list:
        gid_remap_dict[gid] = str(remap_id)
        remap_id += 1
    print('feat size: {}'.format(remap_id))

    with open(remap_dict_file, 'wb') as f:
        pkl.dump(uid_remap_dict, f)
        pkl.dump(iid_remap_dict, f)
        pkl.dump(cid_remap_dict, f)
        pkl.dump(sid_remap_dict, f)
        pkl.dump(bid_remap_dict, f)
        pkl.dump(aid_remap_dict, f)
        pkl.dump(gid_remap_dict, f)
    print('remap ids completed')

    # remap file generate
    item_feat_dict = {}
    user_feat_dict = {}
    # for dummy user
    user_feat_dict['0'] = [0, 0]
    with open(joined_raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for i in range(len(lines)):
            uid, iid, cid, sid, bid, time_stamp, btypeid, aid, gid = lines[i][:-1].split(',')
            uid_remap = uid_remap_dict[uid]
            iid_remap = iid_remap_dict[iid]
            cid_remap = cid_remap_dict[cid]
            sid_remap = sid_remap_dict[sid]
            bid_remap = bid_remap_dict[bid]
            aid_remap = aid_remap_dict[aid]
            gid_remap = gid_remap_dict[gid]
            item_feat_dict[iid_remap] = [int(cid_remap), int(sid_remap), int(bid_remap)]
            user_feat_dict[uid_remap] = [int(aid_remap), int(gid_remap)]
    print('remaped file generated')


def get_season(month):
    if month >= 10:
        return 3
    elif month >= 7 and month <= 9:
        return 2
    elif month >= 4 and month <= 6:
        return 1
    else:
        return 0


def get_ud(day):
    if day <= 15:
        return 0
    else:
        return 1


def remap(raw_file, remap_dict_file, remap_file):
    with open(remap_dict_file, 'rb') as f:
        uid_remap_dict = pkl.load(f)
        iid_remap_dict = pkl.load(f)
        cid_remap_dict = pkl.load(f)
        sid_remap_dict = pkl.load(f)
        bid_remap_dict = pkl.load(f)
        aid_remap_dict = pkl.load(f)
        gid_remap_dict = pkl.load(f)

    newlines = []
    with open(raw_file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            uid, iid, cid, sid, bid, date, btypeid, aid, gid = line[:-1].split(',')
            if btypeid != '0':
                continue

            uid = uid_remap_dict[uid]
            iid = iid_remap_dict[iid]
            cid = cid_remap_dict[cid]
            sid = sid_remap_dict[sid]
            bid = bid_remap_dict[bid]
            aid = aid_remap_dict[aid]
            gid = gid_remap_dict[gid]

            date = '2015' + date
            time_stamp = str(int(time.mktime(datetime.datetime.strptime(date, "%Y%m%d").timetuple())))
            newline = ','.join([uid, aid, gid, iid, cid, sid, bid, time_stamp]) + '\n'
            newlines.append(newline)

    with open(remap_file, 'w') as f:
        f.writelines(newlines)

    delete_if_exists(raw_file)
    delete_if_exists(remap_dict_file)


def sort_log(log_ts_file, sorted_log_ts_file):
    line_dict = {}
    with open(log_ts_file) as f:
        for line in f:
            line_items = line[:-1].split(',')
            uid = line_items[0]
            ts = int(line_items[-1])
            if uid not in line_dict:
                line_dict[uid] = [[line, ts]]
            else:
                line_dict[uid].append([line, ts])

    for uid in line_dict:
        line_dict[uid].sort(key=lambda x: x[1])
    print('sort complete')
    newlines = []
    for uid in line_dict:
        for tup in line_dict[uid]:
            newlines.append(tup[0])
    with open(sorted_log_ts_file, 'w') as f:
        f.writelines(newlines)

    delete_if_exists(log_ts_file)


def gen_samples(user_seq_dict, user_feat_dict, item_set, uid):
    train_list = []
    test_list = []
    aid, gid = user_feat_dict[uid]
    user_seq = user_seq_dict[uid]
    for idx in range(len(user_seq)):
        iid, cid, sid, bid, time_stamp = user_seq[idx]
        item = (iid, cid, sid, bid)
        clk = random.randint(0, 1)
        if clk == 0:
            item = neg_sample(item_set, [x[:-1] for x in user_seq[:idx]], item)
        history = user_seq[:idx][-MAX_LENGTH:]
        iid_his = '^'.join(x[0] for x in history)
        cid_his = '^'.join(x[1] for x in history)
        sid_his = '^'.join(x[2] for x in history)
        bid_his = '^'.join(x[3] for x in history)
        sample = ','.join(
            [str(clk), uid, aid, gid, item[0], item[1], item[2], item[3], iid_his, cid_his, sid_his, bid_his]) + '\n'
        if time_stamp < TEST_THRESHOLD:
            train_list.append(sample)
        else:
            test_list.append(sample)
    return train_list, test_list


def gen_target_seq(input_file,
                   target_train_file,
                   target_vali_file,
                   target_test_file):
    user_seq_dict = {}
    user_feat_dict = {}
    item_set = set()

    with open(input_file, 'r') as f:
        for line in f:
            uid, aid, gid, iid, cid, sid, bid, time_stamp = line[:-1].split(',')
            item_set.add((iid, cid, sid, bid))
            time_stamp = int(time_stamp)
            if uid not in user_feat_dict:
                user_feat_dict[uid] = (aid, gid)
                user_seq_dict[uid] = [(iid, cid, sid, bid, time_stamp)]
            else:
                if user_seq_dict[uid][-1][0] != iid or time_stamp - user_seq_dict[uid][-1][-1] > REMOVE_DUP_THRESHOLD:
                    user_seq_dict[uid].append((iid, cid, sid, bid, time_stamp))

    item_set = list(item_set)

    f_train = open(target_train_file, 'w')
    f_vali = open(target_vali_file, 'w')
    f_test = open(target_test_file, 'w')

    f_train.write(f'{HEADER}\n')
    f_vali.write(f'{HEADER}\n')
    f_test.write(f'{HEADER}\n')

    for uid in tqdm.tqdm(user_seq_dict.keys()):
        if len(user_seq_dict[uid]) > 3:
            train_list, test_list = gen_samples(user_seq_dict, user_feat_dict, item_set, uid)
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
    user_info_format1 = os.path.join(PATH, 'user_info_format1.csv')
    user_log_format1 = os.path.join(PATH, 'user_log_format1.csv')
    joined_user_behavior = os.path.join(PATH, 'joined_user_behavior.csv')
    remap_dict = os.path.join(PATH, 'remap_dict.pkl')
    remap_joined_user_behavior = os.path.join(PATH, 'remap_joined_user_behavior.csv')
    sorted_remap_joined_user_behavior = os.path.join(PATH, 'sorted_remap_joined_user_behavior.csv')
    train_path = os.path.join(PATH, 'train.csv')
    valid_path = os.path.join(PATH, 'valid.csv')
    test_path = os.path.join(PATH, 'test.csv')

    if os.path.exists(sorted_remap_joined_user_behavior):
        gen_target_seq(sorted_remap_joined_user_behavior, train_path, valid_path, test_path)
        return

    join_user_profile(user_info_format1, user_log_format1, joined_user_behavior)
    feateng(joined_user_behavior, remap_dict)
    remap(joined_user_behavior, remap_dict, remap_joined_user_behavior)
    sort_log(remap_joined_user_behavior, sorted_remap_joined_user_behavior)
    gen_target_seq(sorted_remap_joined_user_behavior, train_path, valid_path, test_path)


if __name__ == "__main__":
    main()

