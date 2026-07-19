import os
import random


def delete_if_exists(path):
    if os.path.exists(path):
        os.remove(path)


def neg_sample(item_set, user_seq, item):
    while True:
        i = random.choice(item_set)
        if i != item:
            return i


def truncate_list(l, max_len):
    if len(l) <= max_len:
        return l
    return l[-max_len:]
    # return list(random.sample(l, max_len))


def flatten(l):
    res = []
    for i in l:
        res.extend(i)
    return res
