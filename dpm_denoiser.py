"""
Train and test script for the DPM denoiser.
"""
from DMCE import utils, DiffusionModel, Trainer, Tester, UNet
import os
import os.path as path
import argparse
import modules.utils as ut
import datetime
import csv
import matplotlib.pyplot as plt

CUDA_DEFAULT_ID = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cpu', type=str)

    # get the used device
    args = parser.parse_args()
    device = args.device

    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    num_train_samples = 100_000
    num_val_samples = 10_000
    num_test_samples = 10_000
    seed = 453451
    dataset = 'rand_gmm' # {rand_gmm, MNIST_gmm, FASHION_MNIST_gmm, audio_gmm}
    normalize = True
    if dataset == 'rand_gmm':
        mode = '1D'
        complex_data = False
        dtype_data = 'real'
        n_dim = 64
        n_components = 128
    elif dataset == 'audio_gmm':
        mode = '1D'
        complex_data = True
        n_components = 128
        n_dim = 256
        dtype_data = 'complex'
        normalize = False
    else:
        mode = '2D'
        n_dim = 28**2
        n_components = 128
        complex_data = False
        dtype_data = 'real'

    make_zero_mean = True # shift data to be zero-mean
    norm_per_dim = False # normalize per-entry variance or average variance along the data dimension

    return_all_timesteps = False # return MSEs of all intermediate DPM timesteps

    # Load data
    data_train, data_val, data_test = ut.load_or_create_data_gmm(n_components=n_components, n_dim=n_dim,
                     zeromean=False, num_train_samples=num_train_samples, num_test_samples=num_test_samples,
                     num_val_samples=num_val_samples, seed=seed, return_torch=True, normalize=normalize, dataset=dataset,
                     mode=mode, make_zero_mean=make_zero_mean, norm_per_dim=norm_per_dim, dtype=dtype_data)

    # set data params
    train_dataset = ''
    test_dataset = ''
    cwd = os.getcwd()
    bin_dir = path.join(cwd, 'bin')
    data_shape = tuple(data_train.shape[1:])

    # data parameter dictionary, which is saved in 'sim_params.json'
    data_dict = {
        'bin_dir': str(bin_dir),
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'num_test_samples': num_test_samples,
        'train_dataset': train_dataset,
        'test_dataset': test_dataset,
        'n_antennas': n_dim,
        'mode': mode,
        'data_shape': data_shape,
        'complex_data': complex_data
    }

    # set Diffusion model params
    num_timesteps = 300 # number of timesteps T of DPM
    loss_type = 'l2'
    which_schedule = 'linear' # beta-schedule

    max_snr_dB = 40
    beta_start = 1 - 10**(max_snr_dB/10) / (1 + 10**(max_snr_dB/10))
    if num_timesteps == 5:
        beta_end = 0.95  # -22.5dB
    elif num_timesteps == 10:
        beta_end = 0.7  # -22.5dB
    elif num_timesteps == 50:
        beta_end = 0.2  # -22.5dB
    elif num_timesteps == 100:
        beta_end = 0.1 # -22.5dB
    elif num_timesteps == 300:
        beta_end = 0.035  # -23dB
    elif num_timesteps == 500:
        beta_end = 0.02 #-22dB
    elif num_timesteps == 1_000:
        beta_end = 0.01 #-22dB
    elif num_timesteps == 10_000:
        beta_end = 0.001 #-24dB
    else:
        beta_end = 0.035

    objective = 'pred_noise'  # one of 'pred_noise' (L_n), 'pred_x_0' (L_h), 'pred_post_mean' (L_mu)
    loss_weighting = False # consider pre-factor in loss function or not
    clipping = False # clip data in reverse process, e.g., for images
    reverse_method = 'reverse_mean'  # either 'reverse_mean' or 'ground_truth'
    reverse_add_random = False  # True: Re-sampling method (stochastic) | False: Reverse Mean Forwarding (deterministic)

    # diffusion model parameter dictionary, which is saved in 'sim_params.json'
    diff_model_dict = {
        'data_shape': data_shape,
        'complex_data': complex_data,
        'loss_type': loss_type,
        'which_schedule': which_schedule,
        'num_timesteps': num_timesteps,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'objective': objective,
        'loss_weighting': loss_weighting,
        'clipping': clipping,
        'reverse_method': reverse_method,
        'reverse_add_random': reverse_add_random
    }

    # set UNet params
    ch_data = data_shape[0]
    ch_init = 16 #int(np.random.choice([8, 16, 24, 32]))
    ch_out = ch_data
    if mode == '1D':
        if n_dim >= 128:
            ch_multipliers = (1, 2, 4, 8, 16)
        else:
            ch_multipliers = (1, 2, 4, 8)
    else:
        ch_multipliers = (1, 2, 4)
    num_res_blocks = 2
    kernel_size = 3
    dropout = 0.
    norm_type = 'batch'

    # UNet parameter dictionary, which is saved in 'sim_params.json'
    unet_dict = {
        'ch_data': ch_data,
        'ch_init': ch_init,
        'ch_out': ch_out,
        'kernel_size': kernel_size,
        'mode': mode,
        'ch_multipliers': ch_multipliers,
        'num_res_blocks': num_res_blocks,
        'dropout': dropout,
        'norm_type': norm_type,
        'device': device
    }

    # set Trainer params
    batch_size = 128
    lr_init = ut.rand_exp(1e-5, 1e-3)[0]
    lr_step_multiplier = 1.0 # 0.5
    epochs_until_lr_step = 150 #np.random.randint(50, 600) #150
    num_epochs = 500
    val_every_n_batches = 2000
    num_min_epochs = 50 # minimum training epochs before early stopping is allowed
    num_epochs_no_improve = 20 # number of epochs without val-loss improvement before early stopping
    track_val_loss = True
    track_fid_score = False
    track_mmd = False
    use_fixed_gen_noise = True
    use_ray = False
    save_mode = 'best' # best, newest, all
    dir_result = path.join(cwd, 'results')
    timestamp = utils.get_timestamp()
    dir_result = path.join(dir_result, timestamp)

    # Trainer parameter dictionary, which is saved in 'sim_params.json'
    trainer_dict = {
        'batch_size': batch_size,
        'lr_init': lr_init,
        'lr_step_multiplier': lr_step_multiplier,
        'epochs_until_lr_step': epochs_until_lr_step,
        'num_epochs': num_epochs,
        'val_every_n_batches': val_every_n_batches,
        'track_val_loss': track_val_loss,
        'track_fid_score': track_fid_score,
        'track_mmd': track_mmd,
        'use_fixed_gen_noise': use_fixed_gen_noise,
        'save_mode': save_mode,
        'mode': mode,
        'dir_result': str(dir_result),
        'use_ray': use_ray,
        'complex_data': complex_data,
        'num_min_epochs': num_min_epochs,
        'num_epochs_no_improve': num_epochs_no_improve,
    }

    # set Tester params
    batch_size_test = 512
    criteria = ['nmse']

    # Tester parameter dictionary, which is saved in 'sim_params.json'
    tester_dict = {
        'batch_size': batch_size_test,
        'criteria': criteria,
        'complex_data': complex_data,
        'return_all_timesteps': return_all_timesteps,
    }

    # create result directory
    os.makedirs(dir_result, exist_ok=True)

    # instantiate UNet, DiffusionModel, Trainer and Tester
    unet = UNet(**unet_dict)
    diffusion_model = DiffusionModel(unet, **diff_model_dict)
    trainer = Trainer(diffusion_model, data_train, data_val, **trainer_dict)
    tester = Tester(diffusion_model, data_test, **tester_dict)

    # Print number of trainable parameters
    print(f'Number of trainable model parameters: {diffusion_model.num_parameters}')

    # other parameters dictionary, which is saved in 'sim_params.json'
    misc_dict = {'num_parameters': diffusion_model.num_parameters}

    # save the simulation parameters as a JSON file
    sim_dict = {
        'data_dict': data_dict,
        'diff_model_dict': diff_model_dict,
        'unet_dict': unet_dict,
        'trainer_dict': trainer_dict,
        'tester_dict': tester_dict,
        'misc_dict': misc_dict
    }

    utils.save_params(dir_result=dir_result, filename='sim_params', params=sim_dict)

    # run training routine
    train_dict = trainer.train()
    utils.save_params(dir_result=dir_result, filename='train_results', params=train_dict)

    params = dict()
    params['dim'] = n_dim
    params['components'] = n_components
    params['data_train'] = num_train_samples
    params['data_test'] = num_test_samples
    params['data_val'] = num_val_samples
    params['epochs'] = num_epochs
    params['batch_size'] = batch_size
    params['lr_start'] = lr_init
    params['lr_step_mult'] = lr_step_multiplier
    params['epochs_until_lr_step'] = epochs_until_lr_step
    params['timesteps'] = num_timesteps
    params['beta_start'] = beta_start
    params['beta_end'] = beta_end
    params['snr_low'] = diffusion_model.snrs_db.cpu().detach().numpy()[-1]
    params['snr_high'] = diffusion_model.snrs_db.cpu().detach().numpy()[0]
    params['dataset'] = dataset
    params['schedule'] = which_schedule
    params['ch_multipliers'] = ch_multipliers
    params['num_res_blocks'] = num_res_blocks
    params['kernel_size'] = kernel_size
    params['timestamp'] = timestamp
    params['trained_epochs'] = train_dict['trained_epochs']
    params['num_min_epochs'] = num_min_epochs
    params['num_epochs_no_improve'] = num_epochs_no_improve
    params['loss_weighting'] = loss_weighting
    params['make_zero_mean'] = make_zero_mean
    params['norm_per_dim'] = norm_per_dim
    params['ch_init'] = ch_init
    params['seed'] = seed
    file_name = f'./results/dm_paper/dm_est/{date_time}_{dataset}_dim={n_dim}_valdata={num_val_samples}_comps={n_components}_' \
                f'T={num_timesteps}_params.csv'
    os.makedirs('./results/dm_paper/dm_est/', exist_ok=True)
    with open(file_name, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in params.items():
           writer.writerow([key, value])


    file_name = f'./results/dm_paper/dm_est/{date_time}_{dataset}_dim={n_dim}_valdata={num_val_samples}_comps={n_components}_' \
                f'T={num_timesteps}_loss.png'
    plt.figure()
    plt.semilogy(range(1, len(train_dict['train_losses'])+1), train_dict['train_losses'], label='train-loss')
    plt.semilogy(range(1, len(train_dict['val_losses'])+1), train_dict['val_losses'], label='val-loss')
    plt.legend(['train-loss', 'val-loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(file_name)

    # run testing routine
    test_dict = tester.test()

    if return_all_timesteps:
        # plot all curves
        file_name = f'./results/dm_paper/dm_est/{date_time}_{dataset}_dim={n_dim}_valdata={num_val_samples}_comps={n_components}_' \
                    f'T={num_timesteps}_perstep.png'
        plt.figure()
        lines = []
        for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
            mse_list_allsteps = test_dict[criteria[0]]['NMSEs_total_power'][isnr]
            snr_now = test_dict[criteria[0]]['SNRs'][isnr]
            n_timesteps_eval = len(mse_list_allsteps)
            lines += plt.semilogy(range(num_timesteps-n_timesteps_eval+1, num_timesteps+1), mse_list_allsteps, label=f'SNR = {int(snr_now)}')
            #plt.legend([f'SNR = {int(snr_now)}'])
            plt.xlabel('Timesteps')
            plt.ylabel('nMSE')
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels)
        plt.savefig(file_name)

        # save all mses
        mse_list = list()
        mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
        mse_list[-1].insert(0, 'SNR')
        mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
        mse_list[-1].insert(0, 'nmse_dm')
        mse_list = [list(i) for i in zip(*mse_list)]
        print(mse_list)
        file_name = f'./results/dm_paper/dm_est/{date_time}_{dataset}_dim={n_dim}_valdata={num_val_samples}_comps={n_components}_T={num_timesteps}_perstep.csv'
        with open(file_name, 'w') as myfile:
            wr = csv.writer(myfile, lineterminator='\n')
            wr.writerows(mse_list)

        # remove all mses except last to save it later
        for isnr in range(len(test_dict[criteria[0]]['NMSEs_total_power'])):
            test_dict[criteria[0]]['NMSEs_total_power'][isnr] = test_dict[criteria[0]]['NMSEs_total_power'][isnr][-1]

    mse_list = list()
    mse_list.append(test_dict[criteria[0]]['SNRs'].copy())
    mse_list[-1].insert(0, 'SNR')
    mse_list.append(test_dict[criteria[0]]['NMSEs_total_power'].copy())
    mse_list[-1].insert(0, 'nmse_dm')
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_paper/dm_est/{date_time}_{dataset}_dim={n_dim}_valdata={num_val_samples}_comps={n_components}_T={num_timesteps}.csv'
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)

    utils.save_params(dir_result=dir_result, filename='test_results', params=test_dict)


if __name__ == '__main__':
    main()
