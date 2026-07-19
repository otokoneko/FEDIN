import glob
import json
import os

import yaml

os.chdir(os.path.dirname(os.path.realpath(__file__)))
import logging
from datetime import datetime
from fuxictr.utils import set_logger, print_to_json, load_dataset_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.preprocess import FeatureProcessor, build_dataset
import argparse
import os
import models
from pathlib import Path


def get_parquet_path(params):
    data_root = os.path.join(params['data_root'], params['dataset_id'])
    train_data = os.path.join(data_root, 'train.parquet')
    valid_data = os.path.join(data_root, 'valid.parquet')
    test_data = os.path.join(data_root, 'test.parquet')
    return train_data, valid_data, test_data


def check_parquet(params):
    train_data, valid_data, test_data = get_parquet_path(params)
    return os.path.exists(train_data) and os.path.exists(valid_data) and os.path.exists(test_data)


def load_config(config_dir, experiment_id):
    params = load_model_config(config_dir, experiment_id)
    data_params = load_dataset_config(config_dir, params['dataset_id'])
    params.update(data_params)
    return params


def load_model_config(config_dir, experiment_id):
    model_configs = glob.glob(os.path.join(config_dir, "model_config.yaml"))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, "model_config/*.yaml"))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()
    for config in model_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            found_params.update(config_dict)
    # Update base and exp_id settings consectively to allow overwritting when conflicts exist
    params = found_params.get('Base', {})
    target_params = found_params.get(experiment_id, {})
    if 'base_expid' in target_params:
        params.update(found_params.get(target_params['base_expid'], {}))
    params.update(target_params)
    assert "dataset_id" in params, f'expid={experiment_id} is not valid in config.'
    params["model_id"] = experiment_id
    return params


def print_result(result, metrics):
    out = []
    for key in metrics:
        if key in result:
            out.append('{:.6f}'.format(result[key]))
        else:
            out.append('')
    return out


def load_params(args):
    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    if not args['f'] and check_parquet(params):
        train_data, valid_data, test_data = get_parquet_path(params)
        params["train_data"] = train_data
        params["valid_data"] = valid_data
        params["test_data"] = test_data
        params["data_format"] = 'parquet'
    if args['seed'] != -1:
        params['seed'] = args['seed']
    seed = params['seed']
    params["model_id"] = params["model_id"] + f'_{seed}'
    if 'extra_name' in params:
        extra_name = params['extra_name']
        params["model_id"] += f'_{extra_name}'
    params['gpu'] = args['gpu']
    return experiment_id, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DeepFM_test', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=-1, help='The gpu index, -1 for cpu')
    parser.add_argument('--seed', type=int, default=-1, help='Set random seed.')
    parser.add_argument('--make-dataset', action='store_true', help='Construct dataset.')
    parser.add_argument('--rm', action='store_true', help='Remove model when exiting.')
    parser.add_argument('-q', action='store_true', help='Skip valid and test.')
    parser.add_argument('-f', action='store_true', help='Force rebuild dataset.')
    args = vars(parser.parse_args())

    experiment_id, params = load_params(args)

    set_logger(params)
    logging.info("Params: " + print_to_json(params))

    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        # Build feature_map and transform data
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)

    if args['make_dataset']:
        return

    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(models, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters()  # print number of parameters used in model

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    test_result = {}

    if not args['q']:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
        test_result = model.evaluate(test_gen)

    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        metrics = params['metrics']
        output = [
                     datetime.now().strftime('%Y%m%d %H:%M:%S'),
                     params['model_id'],
                 ] + print_result(test_result, metrics)
        output.append(json.dumps(params))
        line = '\t'.join(map(str, output)) + '\n'
        fw.write(line)

    if args['rm']:
        logging.info('Delete model: ' + model.checkpoint)
        os.remove(model.checkpoint)


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    main()
