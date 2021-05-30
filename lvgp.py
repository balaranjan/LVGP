import numpy as np
from numpy.linalg import inv
from scipy.linalg import cholesky
from scipy.optimize import minimize
import time
from skopt.sampler import Lhs
from skopt.space import Space
import pandas as pd
import warnings
from sklearn.metrics import r2_score, mean_squared_error
from multiprocessing import Pool


def lvgp_kernel(X1, X2, phi_full):

    k = X1.shape[0]
    p = X1.shape[1]
    kk = X2.shape[0]

    R = np.zeros((k, kk))
    phi_full = np.array(phi_full)

    if p != len(phi_full):
        print('Shapes do not match.')
        exit(0)

    if k >= kk:
        for i in range(kk):
            R[:, i] = (np.power(X1.T - X2[i].reshape(-1, 1), 2) * (10 ** phi_full).reshape(-1, 1)).sum(axis=0)
    else:
        for i in range(k):
            R[i, :] = (np.power(X2.T - X1[i].reshape(-1, 1), 2) * (10 ** phi_full).reshape(-1, 1)).sum(axis=0)

    R = np.exp(-0.5 * R)

    return R


def lvgp_to_latent(X_qual, lvs_qual, n_lvs_qual, p_qual, z_vec, dim_z, k):
    """
    Transforms qualitative/categorical variables into latent variables.

    param X_qual Matrix or data frame containing (only) the qualitative/categorical data.
    param lvs_qual List containing levels of each qualitative variable
    param n_lvs_qual Number of levels of each qualitative variable
    param p_qual Number of qualitative variables
    param z_vec Latent variable parameters, i.e., latent variable values for each level of qualitative/categorical variables
    param dim_z Dimensionality of latent variables, usually 1 or 2
    param k Number of data points, equal to \code{nrow(X_qual)}

    return Matrix containing transformed data
    """

    X_qual_la = np.zeros((k, p_qual * dim_z))
    # note: the first levels of each variable are zeros in the latent space,
    # no need to touch upon.

    start = 0
    for i in range(p_qual):
        n_lvs = n_lvs_qual[i]
        lvs = lvs_qual[i]
        end = (dim_z) * (n_lvs - 1) + start

        z_i = z_vec[start: end]
        start = end

        # map
        zstart = 0
        for j in range(1, n_lvs):
            zend = dim_z + zstart
            X_qual_la[X_qual[:, i] == lvs[j], i*dim_z:(i+1)*dim_z] = z_i[zstart:zend]
            zstart = zend

    return X_qual_la


def lvgp_nll(p_quant, p_qual, lvs_qual, n_lvs_qual, dim_z, X_quant, X_qual, Y, min_eig, k, M):
    """
    description Calculates the negative log-likelihood (excluding all the constant terms) as described in \code{reference 1}.

    param hyperparam Hyperparameters of the LVGP model
    param p_quant Number of quantative variables
    param p_qual Number of qualitative variables
    param lvs_qual Levels of each qualitative variable
    param n_lvs_qual Number of levels of each qualitative variable
    param dim_z Dimensionality of latent variables, usually 1 or 2
    param X_quant Input data of quantative variables
    param X_qual Input data of qualitative variables
    param Y Vector containing the outputs of data points
    param min_eig The smallest eigen value that the correlation matrix is allowed to have, which determines the nugget added to the correlation matrix.
    param k Number of data points, \code{nrow(X_quant)} or \code{nrow(X_qual)}
    param M Vector of ones with length \code{k}

    """
    Y = Y.reshape(-1, 1)
    M = M.reshape(-1, 1)

    def nll(hyperparams):
        if p_qual == 0:
            # No qualitative variables
            R = lvgp_kernel(X_quant, X_quant, hyperparams)
        else:
            z_vec = hyperparams[p_quant:]
            X_qual_la = lvgp_to_latent(X_qual, lvs_qual, n_lvs_qual,
                                       p_qual, z_vec, dim_z, k)

            if X_quant is not None:
                X_full = np.hstack([X_quant, X_qual_la])
            else:
                X_full = X_qual_la

            phi_full = hyperparams[:p_quant].tolist()
            phi_full.extend([0 for i in range(p_qual * dim_z)])

            R = lvgp_kernel(X_full, X_full, phi_full)

        R = (R + R.T) / 2  # why?

        raw_min_eig = np.linalg.eigvalsh(R, 'U').min()

        #         raw_min_eig = raw_min_eig.min()
        if raw_min_eig < min_eig:
            R += (np.diag(np.full(k, 1)) * (min_eig - raw_min_eig))

        L = cholesky(R).T
        LinvM = np.linalg.solve(L, M)
        beta_hat = np.dot(LinvM.T, np.linalg.solve(L, Y) / np.sum(LinvM ** 2))
        beta_hat = float(beta_hat)

        temp = np.linalg.solve(L, Y - M*beta_hat)
        sigma2 = np.sum(temp ** 2) / k
        if sigma2 < 1e-300:
            sigma2 = 1e-300

        det_R = np.linalg.det(R)

        if det_R < 1e-300:
            det_R = 1e-300

        n_log_l = k * np.log(sigma2) + np.log(det_R)
        return n_log_l

    return nll


def optim_fun(args):
    x0, obj_fn, kwargs = args
    x_sol_ele = minimize(fun=obj_fn(**kwargs), x0=x0)
    return x_sol_ele


def lvgp_fit(X, Y, ind_qual=None, dim_z=2, eps=np.power(10, np.linspace(-1, -8, 15)),
             lb_phi_ini=-2, ub_phi_ini=2, lb_phi_lat=-8, ub_phi_lat=3,
             lb_z=-3, ub_z=3, n_opt=8, max_iter_ini=100, max_iter_lat=20,
             seed=123, progress=True, parallel=False, noise=False, n_cores=1):
    """
    description Fits a latent-variable Gaussian process (LVGP) model to a dataset as described in \code{reference 1}.
    The input variables can be quantitative or qualitative/categorical or mixed.
    The output variable is quantitative and scalar.

    param X Matrix or data frame containing the inputs of training data points. Each row is a data point.
    param Y Vector containing the outputs of training data points
    param ind_qual Vector containing the indices of columns of qualitative/categorical variables
    param dim_z Dimensionality of latent space, usually 1 or 2 but can be higher
    param eps The vector of smallest eigen values that the correlation matrix is allowed to have, which determines the nugget added to the correlation matrix.
    param lb_phi_ini,ub_phi_ini The initial lower and upper search bounds of the scale/roughness parameters (\code{phi}) of quantitative variables
    param lb_phi_lat,ub_phi_lat The later lower and upper search bounds of the scale/roughness parameters (\code{phi}) of quantitative variables
    param lb_z,ub_z The lower and upper search bounds of the latent parameters (\code{z}) of qualitative variables
    param n_opt The number of times the log-likelihood function is optimized
    param max_iter_ini The maximum number of iterations for each optimization run for largest (first) eps case
    param max_iter_lat The maximum number of iterations for each optimization run for after first eps cases
    param seed An integer for the random number generator. Use this to make the results reproducible.
    param progress The switch determining whether to print function run details
    param parallel The switch determining whether to use parallel computing
    param noise The switch for whether the data are assumed noisy
    """

    np.random.seed(seed)

    if progress:
        print("Checking and preprocessing the inputs...")

    if not (isinstance(X, np.ndarray) or isinstance(X, pd.DataFrame)):
        print('X must be a matrix or a data frame')
        exit(0)

    k = X.shape[0]
    p_all = X.shape[1]

    # Boolean, unsigned integer, signed integer, float, complex.
    _NUMERIC_KINDS = set('buifc')

    if ind_qual is None:  # no qualitative variables
        p_qual = 0
        X_qual = None
        X_quant = X

        if not (np.asarray(X_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_quant))):
            print('All the elements of ind_qual must be finite numbers.')
            exit(0)

        lvs_qual = None
        n_lvs_qual = None
        n_z = 0

    else:
        p_qual = len(ind_qual)
        X_qual = X[:, ind_qual]

        if p_qual == p_all:
            X_quant = None
        else:
            X_quant = X[:, np.array([ii for ii in range(X.shape[1]) if ii not in ind_qual])]
            if not (np.asarray(X_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_quant))):
                print('All the elements of ind_qual must be finite numbers.')
                exit(0)

        lvs_qual = [None] * p_qual
        n_lvs_qual = [0] * p_qual

        for i in range(p_qual):
            _levels = sorted(list(set(X_qual[:, i].tolist())))
            lvs_qual[i] = _levels
            n_lvs_qual[i] = len(_levels)

        n_z = dim_z * (sum(n_lvs_qual)-p_qual)  # number of latent params, first ones are zeros

    if Y is None:
        print('Y must be provided.')

    if k != Y.shape[0]:
        print('The number of rows (i.e., observations) in X and Y should match!')
        exit(0)

    if dim_z not in [1, 2]:
        print('The dimensionality of latent space is uncommon!')

    p_quant = p_all - p_qual

    # normalization of X and Y
    if p_quant > 0:
        X_quant_min = np.min(X_quant, axis=0)  # .reshape(-1, 1)
        X_quant_max = np.max(X_quant, axis=0)  # .reshape(-1, 1)
        X_quant = (X_quant - X_quant_min) / (X_quant_max - X_quant_min)
    else:
        X_quant_min = None
        X_quant_max = None

    Y_min = Y.min()
    Y_max = Y.max()
    Y = (Y - Y_min) / (Y_max - Y_min)

    # initiation for optimization
    n_hyper = p_quant + n_z
    lb_ini = [*[lb_phi_ini for i in range(p_quant)], *[lb_z for i in range(n_z)]]
    ub_ini = [*[ub_phi_ini for i in range(p_quant)], *[ub_z for i in range(n_z)]]
    lb_lat = [*[lb_phi_lat for i in range(p_quant)], *[lb_z for i in range(n_z)]]
    ub_lat = [*[ub_phi_lat for i in range(p_quant)], *[ub_z for i in range(n_z)]]

    if dim_z == 2 and p_qual != 0:
        temp_ind = p_quant - 1
        for i in range(p_qual):

            n_lvs = n_lvs_qual[i]
            lb_ini[temp_ind + dim_z] = -1e-4
            ub_ini[temp_ind + dim_z] = 1e-4
            lb_lat[temp_ind + dim_z] = -1e-4
            ub_lat[temp_ind + dim_z] = 1e-4
            temp_ind += dim_z * n_lvs - 2

    lb_ini = np.array(lb_ini)
    lb_lat = np.array(lb_lat)
    ub_ini = np.array(ub_ini)
    ub_lat = np.array(ub_lat)

    t0 = time.time()

    space = Space([(0., 1.) for i in range(n_hyper)])
    lhs = Lhs(lhs_type="classic", criterion="maximin", iterations=1000)
    A = np.array(lhs.generate(space.dimensions, n_opt))

    hyper0 = (A.T * (ub_ini - lb_ini).reshape(-1, 1) + lb_ini.reshape(-1, 1)).T

    M = np.ones((k, 1))

    # optimization runs
    # only serial

    add_input = {'p_quant': p_quant, 'p_qual': p_qual, 'lvs_qual': lvs_qual, 'n_lvs_qual': n_lvs_qual,
                 'dim_z': dim_z, 'X_quant': X_quant,'X_qual': X_qual, 'Y': Y, 'eps_i': None,
                 'k': k, 'M': M,'lb': lb_ini, 'ub': ub_ini, 'options': {'maxiter': max_iter_ini}}

    n_try = len(eps)
    optim_hist = {}
    optim_hist['i_try'] = []
    optim_hist['hyper0'] = []
    optim_hist['hyper_sol'] = []
    optim_hist['obj_sol'] = []
    optim_hist['flag_sol'] = []

    if progress:
        print('Starting optimization.')

    for i_try in range(n_try):
        eps_i = eps[i_try]
        n_opt_i = len(hyper0)
        hyper_sol = np.zeros((n_opt_i, n_hyper))
        obj_sol = np.zeros((n_opt_i, 1))
        flag_sol = np.zeros((n_opt_i, 1))
        add_input['eps_i'] = eps_i

        if parallel:
            hyper0_list = []
            for hyper_p in hyper0:
                hyper0_list.append([hyper_p, lvgp_nll, add_input])

            pool = Pool(n_cores)
            temp_list = pool.map(optim_fun, hyper0_list)
            print(temp_list)
            exit(0)
        else:

            for j in range(n_opt_i):
                if i_try == 0:
                    # check the nll fn
                    temp = minimize(fun=lvgp_nll(
                        p_quant, p_qual, lvs_qual, n_lvs_qual, dim_z,
                        X_quant, X_qual, Y, eps_i, k, M), x0=hyper0[j].squeeze(),
                        method='L-BFGS-B',
                        bounds=tuple(zip(lb_ini, ub_ini)),
                        options={'maxiter': 100})
                else:
                    temp = minimize(fun=lvgp_nll(
                        p_quant, p_qual, lvs_qual, n_lvs_qual, dim_z,
                        X_quant, X_qual, Y, eps_i, k, M), x0=hyper0[j].squeeze(),
                        method='L-BFGS-B',
                        bounds=tuple(zip(lb_lat, ub_lat)),
                        options={'maxiter': 20})

                hyper_sol[j] = temp.x  # best params
                obj_sol[j] = temp.fun  # loss
                flag_sol[j] = int(temp.success)

            ID = np.argsort(obj_sol.squeeze())[0]
            obj_sol = obj_sol[ID]
            flag_sol = flag_sol[ID]
            hyper_sol = hyper_sol[ID]

        optim_hist['i_try'].append(i_try)
        optim_hist['hyper0'].append(hyper0)
        optim_hist['hyper_sol'].append(hyper_sol)
        optim_hist['obj_sol'].append(obj_sol)
        optim_hist['flag_sol'].append(flag_sol)


    fit_time = time.time() - t0

    # post processing
    if not noise:
        id_best_try = n_try - 1
    else:
        converged = optim_hist['flag_sol'] == 1
        id_best_try = np.argmin(optim_hist['obj_sol'][converged])

    hyper_full = optim_hist['hyper_sol'][id_best_try]
    min_n_log_l = np.min(optim_hist['obj_sol'])

    if p_quant == 0:
        phi = None
    else:
        phi = hyper_full[:p_quant]

    if p_qual == 0:
        z_vec = None
        z = None
    else:
        z_vec = np.array(hyper_full[p_quant:])

    # calc convenient quantities (stored in model$fit_detail)
    if p_qual == 0:
        R = lvgp_kernel(X_quant, X_quant, phi)
        X_full = X_quant
    else:
        X_qual_trans = lvgp_to_latent(X_qual, lvs_qual, n_lvs_qual, p_qual, z_vec, dim_z, k)

        if X_quant is not None:
            X_full = np.hstack([X_quant, X_qual_trans])
            omega = [*phi, *[0. for _ in range(p_qual * dim_z)]]
        else:
            X_full = X_qual_trans
            omega = [0. for _ in range(p_qual * dim_z)]
        R = lvgp_kernel(X_full, X_full, omega)

    R = (R + R.T) / 2

    raw_min_eig = np.linalg.eigvalsh(R, 'U').min(axis=-1)

    if raw_min_eig < eps[id_best_try]:
        nug_opt = eps[id_best_try] - raw_min_eig
        R += (np.diag(np.full(k, 1)) * nug_opt)
    else:
        nug_opt = None

    Y = Y.reshape(-1, 1)
    L = cholesky(R).T

    Linv = np.linalg.solve(L, np.diag(np.full(L.shape[0], 1)))
    LinvM = np.linalg.solve(L, M)
    MTRinvM = np.sum(np.power(LinvM, 2))
    beta_hat = np.dot(LinvM.T, np.linalg.solve(L, Y) / MTRinvM)
    beta_hat = float(beta_hat)

    temp = np.linalg.solve(L, Y - M * beta_hat)
    sigma2 = np.sum(temp**2) / k
    if sigma2 < 1e-300:
        sigma2 = 1e-300

    RinvPYminusMbetaP = np.linalg.solve(L.T, temp)

    # save the fitted model
    model = {}

    model['quantitative_params'] = {
        'phi': phi, 'lb_phi_ini': lb_phi_ini, 'ub_phi_ini': ub_phi_ini,
        'lb_phi_lat': lb_phi_lat, 'ub_phi_lat': ub_phi_lat
    }

    model['qualitative_params'] = {
        'dim_z': dim_z, 'z_vec': z_vec, 'lb_z': lb_z, 'ub_z': ub_z
    }

    model['data'] = {
        'X': X, 'X_quant': X_quant, 'X_qual': X_qual, 'X_full': X_full, 'Y': Y,
        'X_quant_min': X_quant_min, 'X_quant_max': X_quant_max, 'Y_min': Y_min,
        'Y_max': Y_max, 'ind_qual': ind_qual, 'lvs_qual': lvs_qual,
        'n_lvs_qual': n_lvs_qual, 'p_all': p_all, 'p_quant': p_quant,
        'p_qual': p_qual
    }

    model['fit_details'] = {
        'beta_hat': beta_hat, 'sigma2': sigma2, 'MTRinvM': MTRinvM,  'Linv': Linv,
        'LinvM': LinvM, 'RinvPYminusMbetaP': RinvPYminusMbetaP,
        'raw_min_eig': raw_min_eig, 'nug_opt': nug_opt, 'min_n_log_l': min_n_log_l,
        'fit_time': fit_time
    }

    model['settings'] = {
        'max_iter_ini': max_iter_ini, 'max_iter_lat': max_iter_lat, 'seed': seed,
        'n_opt': n_opt, 'lb_phi_ini': lb_phi_ini, 'ub_phi_ini': ub_phi_ini,
        'lb_phi_lat': lb_phi_lat, 'ub_phi_lat': ub_phi_lat, 'lb_z': lb_z,
        'ub_z': ub_z, 'eps': eps
    }

    model['model'] = 'LVGP model'

    return model


def lvgp_predict(X_new, model, MSE_on=False):
    """
    param X_new Matrix or vector containing the input(s) where the predictions are to be made. Each row is an input vector.
    param model The LVGP model fitted by \code{\link[LVGP]{LVGP_fit}}.
    param MSE_on A scalar indicating whether the uncertainty (i.e., mean squared error \code{MSE}) is calculated.
       Set to a non-zero value to calculate \code{MSE}.
    """

    if not ('model' in model and model['model'] == 'LVGP model'):
        print('The 2nd input should be a model of class "LVGP model".')
        exit(0)

    if not isinstance(X_new, np.ndarray):
        print('X_new should be a numpy array.')
        exit(0)

    p_all = model['data']['p_all']

    if X_new.shape[1] != p_all:
        print('The dimensionality of X_new is not correct!')
        exit(0)

    # get params
    p_qual = model['data']['p_qual']
    X_quant_min = model['data']['X_quant_min']
    X_quant_max = model['data']['X_quant_max']
    Y_min = model['data']['Y_min']
    Y_max = model['data']['Y_max']

    X_old_full = model['data']['X_full']
    lvs_qual = model['data']['lvs_qual']
    n_lvs_qual = model['data']['n_lvs_qual']
    ind_qual = model['data']['ind_qual']

    phi = model['quantitative_params']['phi']
    dim_z = model['qualitative_params']['dim_z']
    z_vec = model['qualitative_params']['z_vec']

    beta_hat = model['fit_details']['beta_hat']
    RinvPYminusMbetaP = model['fit_details']['RinvPYminusMbetaP']

    _NUMERIC_KINDS = {'b', 'f', 'i', 'u'}

    # process X_new
    m = X_new.shape[0]

    if p_qual == 0:
        X_new_qual = None
        X_new_quant = X_new

        if not (np.asarray(X_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_new_quant))):
            print('All the elements of X_new must be finite numbers.')
            exit(0)

        X_new_quant = ((X_new_quant.T - X_quant_min.reshape(-1, 1)) / (X_quant_max - X_quant_min).reshape(-1, 1)).T

        R_old_new = lvgp_kernel(X_old_full, X_new_quant, phi)
        R_new_new = lvgp_kernel(X_new_quant, X_new_quant, phi)
    else:
        X_new_qual = X_new[:, ind_qual]

        if p_qual == p_all:
            X_new_quant = None
        else:
            X_new_quant = X_new[:, [ii for ii in range(p_all) if ii not in ind_qual]]
            if not (np.asarray(X_new_quant).dtype.kind in _NUMERIC_KINDS or np.all(np.isfinite(X_new_quant))):
                print('All the elements of X_new must be finite numbers.')
                exit(0)

            X_new_quant = ((X_new_quant.T - X_quant_min.reshape(-1, 1)) / (X_quant_max - X_quant_min).reshape(-1, 1)).T

        X_new_qual_la = lvgp_to_latent(X_new_qual, lvs_qual, n_lvs_qual, p_qual, z_vec, dim_z, m)

        if X_new_quant is not None:
            X_new_full = np.hstack([X_new_quant, X_new_qual_la])
            phi_full = np.array([*phi, *[0. for _ in range(p_qual * dim_z)]])
        else:
            X_new_full = X_new_qual_la
            phi_full = np.array([0. for _ in range(p_qual * dim_z)])

        R_old_new = lvgp_kernel(X_old_full, X_new_full, phi_full)
        R_new_new = lvgp_kernel(X_new_full, X_new_full, phi_full)

    R_new_new = (R_new_new + R_new_new.T) / 2

    # calc predictions
    predictions = {}

    Y_hat = beta_hat + np.dot(R_old_new.T, RinvPYminusMbetaP)
    Y_hat = ((Y_hat.T * (Y_max - Y_min)) + Y_min).T
    predictions['Y_hat'] = Y_hat

    if MSE_on:
        # calculate the uncertainty
        sigma2 = model['fit_details']['sigma2']
        Linv = model['fit_details']['Linv']
        MTRinvM = model['fit_details']['MTRinvM']
        LinvM = model['fit_details']['LinvM']

        temp = np.dot(Linv, R_old_new)
        W = 1 - np.dot(LinvM.T, temp)
        MSE = sigma2 * (R_new_new - np.dot(temp.T, temp) + np.dot(W.T, W)/MTRinvM) * (Y_max-Y_min)**2
        predictions['MSE'] = np.nan_to_num(np.sqrt(np.diag(MSE))) # MSE
    # print('do')
    return predictions


if __name__ == '__main__':
    from scipy.stats import norm

    def ei(mu, sigma, loss_optimum, scaling_factor=1):

        if float(sigma) == 0.0:
            return 0.
        with np.errstate(divide='ignore'):
            Z = scaling_factor * (loss_optimum - mu) / sigma
            expected_improvement = scaling_factor * (loss_optimum - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
            # expected_improvement[sigma == 0.0] = 0.0

            return 1 * expected_improvement


    # import sklearn.gaussian_process as gp
    #
    # kernel = gp.kernels.Matern()
    # model = gp.GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
    #                                     n_restarts_optimizer=10, normalize_y=True,
    #                                     random_state=123)

    # df = pd.read_csv('hoip.csv').drop('Unnamed: 0', axis=1)
    # df10 = pd.read_csv('hoip_train.csv') #.sample(10)
    # # df10 = df[df['BE'] > -30].sample(10)
    # X = df10[['C', 'X1', 'X2', 'X3', 'S']].to_numpy()
    # Y = df10['BE'].to_numpy()
    #
    # model = lvgp_fit(X, Y, ind_qual=[0, 1, 2, 4], n_opt=8, noise=True)
    #
    # dfte = pd.read_csv('hoip_test.csv')
    # X_te = dfte[['C', 'X1', 'X2', 'X3', 'S']].to_numpy()
    # Y_te = dfte['BE'].to_numpy()
    # pred = lvgp_predict(X_te, model, MSE_on=True)
    # MSE = pred['MSE']
    # pred = pred['Y_hat'].reshape(-1)
    #
    # # model.fit(X, Y)
    # # pred, MSE = model.predict(X_te, return_std=True)
    #
    # for i in range(Y_te.shape[0]):
    #     e = ei(pred[i], MSE[i], loss_optimum=np.min(Y))
    #     print(round(Y_te[i], 2), round(pred[i], 2), round(MSE[i], 2), round(e, 3))
    #
    # # print(Y_te.shape, pred.shape)
    # print('RMSE: ', np.sqrt(mean_squared_error(Y_te, pred))/(max(Y_te)-min(Y_te)))
    # print('R2 : ', r2_score(Y_te, pred))
    # print(model['fit_details']['fit_time'])

    df = pd.read_csv('train.csv').drop('Unnamed: 0', axis=1)
    X = df[['V1', 'V2', 'V3']].to_numpy()
    Y = df['V4'].to_numpy()

    dft = pd.read_csv('test.csv').drop('Unnamed: 0', axis=1)
    X_te = dft[['V1', 'V2', 'V3']].to_numpy()
    Y_te = dft['V4'].to_numpy()
    model = lvgp_fit(X, Y, ind_qual=[2], n_opt=8, progress=False, noise=False)
    pred = lvgp_predict(X_te, model, MSE_on=True)
    MSE = pred['MSE']
    pred = pred['Y_hat'].reshape(-1)
    # print(MSE)
    # print(Y_te.shape, pred.shape)
    print('RMSE: ', np.sqrt(mean_squared_error(Y_te, pred))/(max(Y_te)-min(Y_te)))
    print('R2 : ', r2_score(Y_te, pred))

