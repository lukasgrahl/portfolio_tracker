from src.toml import load_toml
from src.pull_data import load_data
from src.filter import get_kalman_cv, run_kalman_filter
from src.utils import is_outlier
from src.hmm import run_hmm
import os

from settings import PROJECT_ROOT
import view


config = load_toml(os.path.join(PROJECT_ROOT, 'config.toml'))
default_pull_start_date = config['default_values']['pull_start_date']
default_KF_cv_samples = config['default_values']['kf_cv_samples']
default_KF_analysis_time = config['default_values']['kf_analysis_time']
default_KF_measurement_noise = config['default_values']['kf_measurement_noise']
default_HMM_no_states = config['default_values']['hmm_no_states']
default_HMM_cv_samples = config['default_values']['hmm_cv_samples']
default_HMM_start_init = config['default_values']['hmm_start_init']
default_HMM_cv_sample_sizes = config['default_values']['hmm_cv_sample_sizes']
# get data related vals
train_test_size = config['data']['train_test_size']
outlier_interval = config['data']['outlier_std_interval']

if __name__ == "__main__":
    #### ST Set Up ####
    SEL_IND, SEL_IND_TICKER, PULL_START_DATE, PULL_END_DATE = view.set_up_page()

    DF_PRICES, DF_RETS, SEL_IND_NLARGEST_TICKERS, LEAD_NAME = load_data(SEL_IND, SEL_IND_TICKER,
                                                                        str(PULL_START_DATE), str(PULL_END_DATE),
                                                                        no_internet=True)
    hmm_states, cv_samples, hmm_init, dt_start, measurement_noise, analysis_time, cv_samples_kalman = view.set_up_sliders()

    #### Kalman Filter ####
    # cross validate kalman filter
    kf_conf_mat, kf_roc_score = get_kalman_cv(data=DF_RETS, endog=[LEAD_NAME], exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                              measurement_noise=measurement_noise,
                                              cv_index_len=DF_PRICES.loc[PULL_START_DATE: dt_start].shape[0],
                                              sample_len_weeks=analysis_time, no_samples=cv_samples_kalman)
    # run filter on test data
    kf_xtrue, kf_xpred, kf_xfilt = run_kalman_filter(endog=[LEAD_NAME], exog=SEL_IND_NLARGEST_TICKERS.copy(),
                                                     data=DF_RETS.loc[str(dt_start): str(PULL_END_DATE)].copy(),
                                                     measurement_noise=measurement_noise)

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
                        hmm_cv_states, hmm_xtrain, hmm_xtest, hmm_train_states, hmm_test_states)
