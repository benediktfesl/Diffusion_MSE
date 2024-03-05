import numpy as np
import datetime
import csv
from modules import utils as ut
import copy
import os


def main():
    date_time_now = datetime.datetime.now()
    date_time = date_time_now.strftime('%Y-%m-%d_%H-%M-%S')  # convert to str compatible with all OSs

    n_dim = 64
    num_train_samples = 100_000
    num_test_samples = 10_000
    num_val_samples = 10_000
    seed = 453451
    rng = np.random.default_rng(seed=seed)

    n_components = 128 # {number of ground-truth GMM components}
    make_zero_mean = True # shift data to be zero-mean
    norm_per_dim = False # normalize per-entry variance or average variance along the data dimension

    dataset = 'rand_gmm' # {rand_gmm, MNIST_gmm, FASHION_MNIST_gmm} ; NOT audio_gmm (since its complex-valued, see audio_gmm.py)

    _, _, data_test, gmm = ut.load_or_create_data_gmm(n_components=n_components, n_dim=n_dim,
                                                 zeromean=False, num_train_samples=num_train_samples,
                                                 num_test_samples=num_test_samples, num_val_samples=num_val_samples,
                                                 seed=seed, return_gmm=True, dataset=dataset, normalize=True,
                                                 make_zero_mean=make_zero_mean, norm_per_dim=norm_per_dim)
    # update n_dim in case of loading generic datasets
    n_dim = data_test.shape[-1]

    snrs = [-20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    mse_list = list()
    snrs_ = snrs.copy()
    snrs_.insert(0, 'SNR')
    mse_list.append(snrs_)

    # compute ground-truth CME
    mse_list.append(['genie-cme'])
    for snr in snrs:
        x_est = np.zeros_like(data_test)
        sigma2 = 10 ** (-snr / 10)
        y_val = data_test + np.sqrt(sigma2) * rng.standard_normal(data_test.shape)
        gmm_y = copy.deepcopy(gmm)
        Cy_inv = ut.prepare_for_prediction(gmm_y, sigma2)
        proba = gmm_y.predict_proba(y_val)
        for i in range(data_test.shape[0]):
            for argproba in range(proba.shape[1]):
                x_est[i] += proba[i, argproba] * ut.lmmse_formula(gmm, Cy_inv, argproba, y_val[i])
        mse_list[-1].append(np.sum(np.abs(data_test - x_est)**2) / np.sum(np.abs(data_test)**2))

    # compute least squares
    mse_list.append(['ls'])
    for snr in snrs:
        sigma2 = 10 ** (-snr / 10)
        y_val = data_test + np.sqrt(sigma2) * rng.standard_normal(data_test.shape)
        mse_list[-1].append(np.sum(np.abs(data_test - y_val) ** 2) / np.sum(np.abs(data_test)**2))

    # print and save MSEs
    mse_list = [list(i) for i in zip(*mse_list)]
    print(mse_list)
    file_name = f'./results/dm_paper/genie-cme/{date_time}_dim={n_dim}_testdata={num_test_samples}' \
                f'_make0mean={make_zero_mean}_normdim={norm_per_dim}_comps={n_components}_{dataset}.csv'
    os.makedirs('./results/dm_paper/genie-cme/', exist_ok=True)
    with open(file_name, 'w') as myfile:
        wr = csv.writer(myfile, lineterminator='\n')
        wr.writerows(mse_list)


if __name__ == '__main__':
    main()