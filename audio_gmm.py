import numpy as np
from modules import utils as ut
import datetime
import csv
import os


def main():
    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')
    dataset = 'audio_gmm'

    seed = 453451
    n_components = 128 # number of GMM components
    n_dim = 256 # number of dimensions
    n_train = 100_000
    n_test = 10_000
    n_val = 10_000
    snrs = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

    # Load data
    _, _, data_test, gmm_audio = ut.load_or_create_data_gmm(n_components=n_components, n_dim=n_dim,
                                                                 zeromean=False,
                                                                 num_train_samples=n_train,
                                                                 num_test_samples=n_test,
                                                                 num_val_samples=n_val, seed=seed, return_gmm=True,
                                                                 return_torch=False, normalize=False, dataset=dataset,
                                                                 mode='1D', make_zero_mean=True,
                                                                 norm_per_dim=False, dtype='complex')
    n_dim = data_test.shape[-1]

    mse_list = list()
    snrs_ = snrs.copy()
    snrs_.insert(0, 'SNR')
    mse_list.append(snrs_)

    mse_list.append(['genie-cme'])
    for snr in snrs:
        sigma2 = 10 ** (-snr / 10)
        y_test = data_test + np.sqrt(sigma2) * ut.crandn(*data_test.shape)
        x_est = gmm_audio.estimate_from_y(y_test, snr, n_dim, n_summands_or_proba='all')
        mse_list[-1].append(np.sum(np.abs(data_test - x_est) ** 2) / np.sum(np.abs(data_test) ** 2))


    mse_list.append(['ls'])
    for snr in snrs:
        sigma2 = 10 ** (-snr / 10)
        y_test = data_test + np.sqrt(sigma2) * ut.crandn(*data_test.shape)
        mse_list[-1].append(np.sum(np.abs(data_test - y_test) ** 2) / np.sum(np.abs(data_test) ** 2))

    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    os.makedirs('./results/dm_paper/genie-cme/', exist_ok=True)
    file_name = f'./results/dm_paper/genie-cme/{date_time}_dim={n_dim}_testdata={n_test}_comps={n_components}_{dataset}.csv'

    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)


if __name__ == '__main__':
    main()