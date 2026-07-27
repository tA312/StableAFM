import csv
import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
from utils.utils_sr_metrics import (
    calculate_decimation_artifact_metrics,
    hard_project_decimation,
)

from data.select_dataset import define_Dataset
from data.sampler_modality import DistributedModalitySampler
from models.select_model import define_Model


VALIDATION_METRIC_ORDER = (
    'psnr', 'ssim', 'mae', 'dc_mae', 'grid4', 'grid8', 'grid16', 'grid32',
    'phase_contrast_mae', 'phase_ratio', 'target_phase_ratio', 'phase_ratio_error',
    'projected_psnr', 'projected_ssim', 'projected_mae', 'projected_dc_mae',
    'projected_grid4', 'projected_grid8', 'projected_grid16', 'projected_grid32',
    'projected_phase_contrast_mae', 'projected_phase_ratio',
    'projected_phase_ratio_error',
)


def batch_value(batch, key, default=None):
    value = batch.get(key, default)
    if torch.is_tensor(value):
        return value.flatten()[0].item()
    if isinstance(value, (list, tuple)):
        return value[0]
    return value


def update_metric_totals(totals, metrics):
    totals['count'] = totals.get('count', 0) + 1
    for key, value in metrics.items():
        totals[key] = totals.get(key, 0.0) + float(value)


def average_metric_totals(totals):
    count = int(totals['count'])
    if count <= 0:
        raise ValueError('Cannot average an empty validation group.')
    return {
        key: value / count
        for key, value in totals.items()
        if key != 'count'
    }


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes'):
        return True
    if value in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


'''
# --------------------------------------------
# training code for MSRResNet
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
# github: https://github.com/cszn/KAIR
# --------------------------------------------
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', dest='local_rank', type=int, default=0)
    parser.add_argument('--dist', type=str2bool, nargs='?', const=True, default=None)

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    if args.dist is not None:
        opt['dist'] = args.dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(
        opt['path']['models'], net_type='G', pretrained_path=opt['path']['pretrained_netG']
    )
    init_iter_E, init_path_E = option.find_last_checkpoint(
        opt['path']['models'], net_type='E', pretrained_path=opt['path']['pretrained_netE']
    )
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(
        opt['path']['models'], net_type='optimizerG',
        pretrained_path=opt['path'].get('pretrained_optimizerG')
    )
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            modality_batch_counts = dataset_opt['dataloader_modality_batch_counts']
            if modality_batch_counts:
                modality_batch_counts = {
                    str(group): int(count)
                    for group, count in modality_batch_counts.items()
                }
                if sum(modality_batch_counts.values()) != dataset_opt['dataloader_batch_size']:
                    raise ValueError(
                        'dataloader_modality_batch_counts must sum to dataloader_batch_size.'
                    )
            persistent_workers = bool(dataset_opt['dataloader_persistent_workers'])
            loader_generator = None
            if dataset_opt['dataloader_seed'] is not None:
                loader_generator = torch.Generator()
                loader_generator.manual_seed(int(dataset_opt['dataloader_seed']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                num_workers = dataset_opt['dataloader_num_workers']//opt['num_gpu']
                if modality_batch_counts:
                    if any(count % opt['num_gpu'] for count in modality_batch_counts.values()):
                        raise ValueError(
                            'Each global modality count must be divisible by the GPU count.'
                        )
                    local_modality_counts = {
                        group: count // opt['num_gpu']
                        for group, count in modality_batch_counts.items()
                    }
                    train_sampler = DistributedModalitySampler(
                        train_set,
                        local_modality_counts,
                        num_replicas=opt['num_gpu'],
                        rank=opt['rank'],
                        shuffle=dataset_opt['dataloader_shuffle'],
                        seed=int(dataset_opt['dataloader_seed'] or seed),
                    )
                else:
                    train_sampler = DistributedSampler(
                        train_set, shuffle=dataset_opt['dataloader_shuffle'],
                        drop_last=True, seed=seed
                    )
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=num_workers,
                                          drop_last=True,
                                          pin_memory=True,
                                          persistent_workers=persistent_workers and num_workers > 0,
                                          generator=loader_generator,
                                          sampler=train_sampler)
            else:
                num_workers = dataset_opt['dataloader_num_workers']
                train_sampler = None
                if modality_batch_counts:
                    train_sampler = DistributedModalitySampler(
                        train_set,
                        modality_batch_counts,
                        shuffle=dataset_opt['dataloader_shuffle'],
                        seed=int(dataset_opt['dataloader_seed'] or seed),
                    )
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=(
                                              dataset_opt['dataloader_shuffle']
                                              if train_sampler is None else False
                                          ),
                                          num_workers=num_workers,
                                          drop_last=True,
                                          pin_memory=True,
                                          persistent_workers=persistent_workers and num_workers > 0,
                                          generator=loader_generator,
                                          sampler=train_sampler)

        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    total_iter = opt['train']['total_iter'] if opt['train']['total_iter'] else 0
    early_stop_patience = opt['train']['early_stop_patience'] if opt['train']['early_stop_patience'] else 0
    early_stop_min_delta = opt['train']['early_stop_min_delta'] if opt['train']['early_stop_min_delta'] else 0
    save_test_images = opt['train']['save_test_images']
    save_test_images = True if save_test_images is None else bool(save_test_images)
    save_preview_only = bool(opt['train']['save_preview_only'])
    report_artifact_metrics = bool(opt['train']['report_artifact_metrics'])
    evaluate_hard_projection = bool(opt['train']['evaluate_hard_projection'])
    metrics_csv_path = os.path.join(opt['path']['task'], 'validation_metrics.csv')
    best_score = -float('inf')
    best_iter = 0
    no_improve_tests = 0
    stop_training = False

    if opt['rank'] == 0 and total_iter > 0:
        logger.info('Training will stop at iter {:,d}.'.format(total_iter))

    for epoch in range(1000000):  # bounded by total_iter/early stopping
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):
            if total_iter > 0 and current_step >= total_iter:
                stop_training = True
                break

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                validation_stop = False

                if opt['rank'] == 0:
                    overall_totals = {}
                    group_totals = {}
                    idx = 0

                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, _ = os.path.splitext(image_name_ext)
                        group = str(batch_value(test_data, 'group', 'default'))
                        sample_id = str(batch_value(test_data, 'sample_id', img_name))

                        model.feed_data(test_data)
                        model.test()

                        visuals = model.current_visuals()
                        bit_depth = int(batch_value(test_data, 'bit_depth', 8))
                        if bit_depth == 16:
                            E_img = util.tensor2uint16(visuals['E'])
                            H_img = util.tensor2uint16(visuals['H'])
                            metric_data_range = 65535.0
                        else:
                            E_img = util.tensor2uint(visuals['E'])
                            H_img = util.tensor2uint(visuals['H'])
                            metric_data_range = 255.0

                        # -----------------------
                        # save estimated image E
                        # -----------------------
                        is_preview = bool(batch_value(test_data, 'save_preview', False))
                        if save_test_images and (not save_preview_only or is_preview):
                            img_dir = os.path.join(opt['path']['images'], sample_id)
                            util.mkdir(img_dir)
                            save_img_path = os.path.join(
                                img_dir, '{:s}_{:d}.png'.format(sample_id, current_step)
                            )
                            util.imsave(E_img, save_img_path)

                        metrics = {
                            'psnr': util.calculate_psnr(
                                E_img, H_img, border=border, data_range=metric_data_range
                            ),
                            'ssim': util.calculate_ssim(
                                E_img, H_img, border=border, data_range=metric_data_range
                            ),
                            'mae': torch.mean(torch.abs(visuals['E'] - visuals['H'])).item(),
                        }

                        if report_artifact_metrics:
                            artifact_metrics = calculate_decimation_artifact_metrics(
                                visuals['E'], visuals['H'], visuals['L'], opt['scale']
                            )
                            metrics['dc_mae'] = artifact_metrics['dc_mae']
                            for key in (
                                'phase_contrast_mae', 'phase_ratio',
                                'target_phase_ratio', 'phase_ratio_error'
                            ):
                                metrics[key] = artifact_metrics[key]
                            for period in (4, 8, 16, 32):
                                metrics['grid{:d}'.format(period)] = artifact_metrics['grid_excess'][period]

                            if evaluate_hard_projection:
                                projected = hard_project_decimation(
                                    visuals['E'], visuals['L'], opt['scale']
                                )
                                projected_img = (
                                    util.tensor2uint16(projected)
                                    if bit_depth == 16 else util.tensor2uint(projected)
                                )
                                metrics['projected_psnr'] = util.calculate_psnr(
                                    projected_img, H_img, border=border,
                                    data_range=metric_data_range
                                )
                                metrics['projected_ssim'] = util.calculate_ssim(
                                    projected_img, H_img, border=border,
                                    data_range=metric_data_range
                                )
                                metrics['projected_mae'] = torch.mean(
                                    torch.abs(projected - visuals['H'])
                                ).item()
                                projected_metrics = calculate_decimation_artifact_metrics(
                                    projected, visuals['H'], visuals['L'], opt['scale']
                                )
                                metrics['projected_dc_mae'] = projected_metrics['dc_mae']
                                for key in (
                                    'phase_contrast_mae', 'phase_ratio', 'phase_ratio_error'
                                ):
                                    metrics['projected_' + key] = projected_metrics[key]
                                for period in (4, 8, 16, 32):
                                    metrics['projected_grid{:d}'.format(period)] = (
                                        projected_metrics['grid_excess'][period]
                                    )

                        update_metric_totals(overall_totals, metrics)
                        update_metric_totals(group_totals.setdefault(group, {}), metrics)
                        logger.info(
                            '{:->4d}--> {:>18s} [{}] | {:<4.2f}dB'.format(
                                idx, sample_id, group, metrics['psnr']
                            )
                        )

                    overall_avg = average_metric_totals(overall_totals)
                    group_averages = {
                        group: average_metric_totals(totals)
                        for group, totals in sorted(group_totals.items())
                    }
                    avg_psnr = overall_avg['psnr']
                    selection_score = avg_psnr

                    # testing log
                    logger.info(
                        '<epoch:{:3d}, iter:{:8,d}, Average PSNR: {:.4f}dB, '
                        'SSIM: {:.6f}, MAE: {:.6e}>'.format(
                            epoch, current_step, avg_psnr,
                            overall_avg['ssim'], overall_avg['mae']
                        )
                    )
                    for group, averages in group_averages.items():
                        logger.info(
                            'Validation group {}: count={}, PSNR={:.4f}dB, '
                            'SSIM={:.6f}, MAE={:.6e}'.format(
                                group, int(group_totals[group]['count']), averages['psnr'],
                                averages['ssim'], averages['mae']
                            )
                        )

                    if report_artifact_metrics:
                        logger.info(
                            'Artifact metrics: DC_MAE={:.6e}, '
                            'GRID4={:+.6f}, GRID8={:+.6f}, GRID16={:+.6f}, GRID32={:+.6f}, '
                            'PHASE_MAE={:.6e}, PHASE_RATIO={:.4f} (target={:.4f}, error={:.4f})'.format(
                                overall_avg['dc_mae'], overall_avg['grid4'],
                                overall_avg['grid8'], overall_avg['grid16'],
                                overall_avg['grid32'], overall_avg['phase_contrast_mae'],
                                overall_avg['phase_ratio'], overall_avg['target_phase_ratio'],
                                overall_avg['phase_ratio_error'],
                            )
                        )

                        if evaluate_hard_projection:
                            logger.info(
                                'Projected metrics: PSNR={:.4f}, SSIM={:.6f}, MAE={:.6e}, DC_MAE={:.6e}, '
                                'GRID4={:+.6f}, GRID8={:+.6f}, GRID16={:+.6f}, GRID32={:+.6f}, '
                                'PHASE_MAE={:.6e}, PHASE_RATIO={:.4f}, PHASE_ERROR={:.4f}'.format(
                                    overall_avg['projected_psnr'], overall_avg['projected_ssim'],
                                    overall_avg['projected_mae'], overall_avg['projected_dc_mae'],
                                    overall_avg['projected_grid4'], overall_avg['projected_grid8'],
                                    overall_avg['projected_grid16'], overall_avg['projected_grid32'],
                                    overall_avg['projected_phase_contrast_mae'],
                                    overall_avg['projected_phase_ratio'],
                                    overall_avg['projected_phase_ratio_error'],
                                )
                            )

                    row = {'iter': current_step, 'count': int(overall_totals['count'])}
                    for key in VALIDATION_METRIC_ORDER:
                        if key in overall_avg:
                            row[key] = overall_avg[key]
                    for group, averages in group_averages.items():
                        if group == 'default':
                            continue
                        row['{}_count'.format(group)] = int(group_totals[group]['count'])
                        for key in VALIDATION_METRIC_ORDER:
                            if key in averages:
                                row['{}_{}'.format(group, key)] = averages[key]
                    write_header = not os.path.isfile(metrics_csv_path)
                    with open(metrics_csv_path, 'a', newline='') as csv_file:
                        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)

                    if selection_score > best_score + early_stop_min_delta:
                        best_score = selection_score
                        best_iter = current_step
                        no_improve_tests = 0
                        logger.info(
                            'New best PSNR: {:.4f} at iter {:,d}. Saving best model.'.format(
                                best_score, best_iter
                            )
                        )
                        model.save('best')
                    else:
                        no_improve_tests += 1
                        logger.info(
                            'No PSNR improvement for {:d} validation(s). '
                            'Current={:.4f}; best={:.4f} at iter {:,d}.'.format(
                                no_improve_tests, selection_score, best_score, best_iter
                            )
                        )
                        if early_stop_patience > 0 and no_improve_tests >= early_stop_patience:
                            logger.info('Early stopping at iter {:,d}: no improvement for {:d} validation(s).'.format(
                                current_step, no_improve_tests
                            ))
                            validation_stop = True

                if opt['dist'] and torch.distributed.is_initialized():
                    stop_tensor = torch.tensor(
                        [1 if validation_stop else 0],
                        device=torch.cuda.current_device()
                    )
                    torch.distributed.broadcast(stop_tensor, src=0)
                    validation_stop = bool(stop_tensor.item())

                if validation_stop:
                    stop_training = True
                    break

        if stop_training:
            if opt['rank'] == 0:
                logger.info('Saving final model at iter {:,d}.'.format(current_step))
                model.save(current_step)
            break

if __name__ == '__main__':
    main()
