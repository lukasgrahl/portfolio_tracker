from src.toml import load_toml
from src.pull_data import load_data
from src.filter import get_kalman_cv, run_kalman_filter
from src.arma import grid_search_arma
from src.utils import is_outlier
from src.hmm import run_hmm
import os
from settings import PROJECT_ROOT
import view


def main():

    config = load_toml(os.path.join(PROJECT_ROOT, 'config.toml'))
    default_HMM_cv_sample_sizes = config['default_values']['hmm_cv_sample_sizes']
    train_test_size = config['data']['train_test_size']
    outlier_interval = config['data']['outlier_std_interval']

    #### ST Set Up ####
    SEL_IND, SEL_IND_TICKER, PULL_START_DATE, PULL_END_DATE = view.set_up_page()

    DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data(SEL_IND, SEL_IND_TICKER,
                                                                        str(PULL_START_DATE), str(PULL_END_DATE),
                                                                        n_largest=config['data']['n_largest_composits'])

    _ = view.set_up_sliders()
    hmm_states, cv_samples, hmm_init, test_sample_start, measurement_noise, analysis_time, cv_samples_kalman = _

    #### ARMA ####
    from datetime import timedelta
    from src.arma import get_ARMA
    ind = (test_sample_start - timedelta(config['default_values']['arma_analaysis_time_days']))
    p, q, d, ma_residuals, arma_params, arma_mod = grid_search_arma(config['data']['p_max'], config['data']['q_max'],
                                                                    DF_RETS.loc[ind: test_sample_start],
                                                                    endog=[LEAD_NAME],
                                                                    exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                                                    sup_warnings=True)

    _, _, _, _, _, arma_mod = get_ARMA(p, q, DF_RETS.loc[test_sample_start:].iloc[:-1], endog=[LEAD_NAME],
                                       exog=SEL_IND_NLARGEST_TICKERS.copy(), sup_warnings=True, vals_only=True)
    arma_predict = arma_mod.predict()
    arma_true = DF_RETS.loc[test_sample_start:, ].iloc[:-1][SEL_IND_TICKER].values.reshape(-1)

    #### Kalman Filter ####
    # cross validate kalman filter
    kf_conf_mat, kf_roc_score = get_kalman_cv(data=DF_RETS.loc[:test_sample_start], endog=[LEAD_NAME],
                                              exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                              measurement_noise=measurement_noise,
                                              cv_index_len=DF_PRICES.loc[:test_sample_start].shape[0],
                                              sample_len_weeks=analysis_time, no_samples=cv_samples_kalman, p=p,
                                              q=q, d=d, ma_resid=ma_residuals, arma_params=arma_params)
    # run filter on test data
    kf_xtrue, kf_xpred, kf_xfilt = run_kalman_filter(endog=[LEAD_NAME], exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                                     data=DF_RETS.loc[test_sample_start: PULL_END_DATE].copy(),
                                                     measurement_noise=measurement_noise, p=p, q=q, d=d,
                                                     ma_resid=ma_residuals, arma_params=arma_params)

    #### HMM ####
    # get data for CV
    data = DF_RETS.drop([item for item in DF_RETS.columns if SEL_IND_TICKER[0] in item], axis=1).copy()
    data = data.join(DF_RETS[[SEL_IND_TICKER[0], LEAD_NAME]])
    data = data.join(DF_PRICES[[f'{SEL_IND_TICKER[0]}_{item}' for item in ['High', 'Low', 'Open', 'Volume']]])
    data = data.join(DF_PRICES[SEL_IND_TICKER].rename(columns={SEL_IND_TICKER[0]: f'{SEL_IND_TICKER[0]}_price'}))
    # outlier selection
    mask = is_outlier(data[LEAD_NAME], outlier_interval)
    data = data[~mask]
    # run hmm
    run_out = run_hmm(data=data, sel_ind_ticker=SEL_IND_TICKER, lead_name=LEAD_NAME,
                      sel_ind_nlargest_ticker=SEL_IND_NLARGEST_TICKERS, hmm_states=hmm_states, hmm_init=hmm_init,
                      cv_samples=cv_samples, train_test_size=train_test_size,
                      cv_sample_sizes=tuple(default_HMM_cv_sample_sizes))
    # get hmm output
    mod, train_cv_states, hmm_cv_states, hmm_cv_statesg, test, train, train_cv = run_out
    _, _, hmm_xtest, hmm_test_states = test
    _, _, hmm_xtrain, hmm_train_states = train
    X_train_cv, y_train_cv = train_cv

    # PLOT
    view.st_plot_output(DF_PRICES, kf_xtrue, kf_xpred, kf_xfilt, kf_conf_mat, kf_roc_score, hmm_cv_statesg,
                        hmm_cv_states, hmm_xtrain, hmm_xtest, hmm_train_states, hmm_test_states, arma_true,
                        arma_predict)

    pass


if __name__ == "__main__":
    main()
