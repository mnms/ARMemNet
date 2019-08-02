import numpy as np
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import os


# Plot results for ONE cell
# If several cells are used in "test_mem_model.py" for testing,
# it should be separated by each cell before running this script.
# To test ONE cell, see "test_cell_ids" in "AR_mem/config.py"

path = './AR_mem/results/20190719-00/'
enb_id = 1111   # For title of plot
cell_id = 1111  # For title of plot
col_list=['CQI', 'RSRP', 'RSRQ', 'DL_PRB_USAGE_RATE', 'SINR', 'UE_TX_POWER', 'PHR', 'UE_CONN_TOT_CNT']
ylim=(-1., 1.) ## range
fixed_ylim=True
threshold=.05
figsize=(12, 24)
style='default'
error_type='mae'

mpl.style.use(style)

error_dict = {
    'mse': lambda pred_y_t, real_y_t: (
        ((pred_y_t - real_y_t) ** 2).mean()
    ),
    'rmse': lambda pred_y_t, real_y_t: np.sqrt(
        ((pred_y_t - real_y_t) ** 2).mean()
    ),
    'mae': lambda pred_y_t, real_y_t: (
        np.abs(pred_y_t - real_y_t).mean()
    ),
    'mape': lambda pred_y_t, real_y_t: np.mean(
        np.abs((real_y_t - pred_y_t) / real_y_t)
    ),
    'mase': lambda pred_y_t, real_y_t: np.mean(
        np.abs(pred_y_t - real_y_t)  /
        np.abs(np.diff(real_y_t, n=1)).mean()
    ),
    'maspe': lambda pred_y_t, real_y_t, real_y_overall: np.mean(
        np.abs(pred_y_t - real_y_t) /
        np.abs(np.diff(real_y_overall, n=1)).mean()
    ),
    'maae': lambda pred_y_t, real_y_t: np.mean(
        np.abs((real_y_t - pred_y_t) /
        np.abs(np.max(pred_y_t) - np.min(pred_y_t)))
    ),
}

colormap_dict = {
        'as_is': '#FF8C00', # real
        'to_be': "#4A708B", # pred
}

pred_y_g = np.load(os.path.join(path, 'pred.npy'))
real_y_g = np.load(os.path.join(path, 'test_y.npy'))
dt = np.load(os.path.join(path, 'test_dt.npy'))

# Scaling for fair comparison with seq2seq model
# def scale(series):
#     series = (series-(-.7))/(1.-(-.7))
#     return series

# pred_y = scale(pred_y_g)
# real_y = scale(real_y_g)

real_y = pred_y_g
pred_y = real_y_g

real_y_swap = np.swapaxes(real_y, 0, 1) # [8, N]
pred_y_swap = np.swapaxes(pred_y, 0, 1) # [8, N]
real_y_swap_g = np.swapaxes(real_y_g, 0, 1) # [8, N]
pred_y_swap_g = np.swapaxes(pred_y_g, 0, 1) # [8, N]

# 8, N
kpi_num_y, seq_length_y = pred_y_swap.shape
assert len(col_list) == kpi_num_y

span_color = np.array(plt.cm.Set2(0))
fig, ax = plt.subplots(
    nrows=kpi_num_y, # 8
    ncols=1,
    figsize=figsize,
)

for kpi_idx in range(kpi_num_y):

    if error_type in ('mape', 'maspe', 'maae'):
        continue

    else:
        err = error_dict[error_type](
            pred_y_swap[kpi_idx], real_y_swap[kpi_idx]
        )
        kpi_name = col_list[kpi_idx]
        kpi_raw_err = err
        kpi_score = err * 100

    ax[kpi_idx].set_title(
        '[{:^15s}]   '.format(kpi_name) +
        'Error:  {:.5f}   '.format(kpi_raw_err) +
        'Score:  {:.2f}   '.format(kpi_score)
    )
    if fixed_ylim:
        ax[kpi_idx].set_ylim(ylim)

    ax[kpi_idx].plot(
#         dt,
        real_y_swap_g[kpi_idx],
        label='real',
        color=colormap_dict['as_is'],
        alpha=.7,
        linewidth=.7,
    )
    ax[kpi_idx].plot(
#         dt,
        pred_y_swap_g[kpi_idx],
        label='pred',
        color=colormap_dict['to_be'],
        linewidth=.8,
    )
    pred_kpi_threshold = threshold
    ax[kpi_idx].fill_between(
        range(real_y_swap_g.shape[1]),
        pred_y_swap_g[kpi_idx] - (pred_y_swap_g[kpi_idx] * pred_kpi_threshold),
        pred_y_swap_g[kpi_idx] + (pred_y_swap_g[kpi_idx] * pred_kpi_threshold),
        color=colormap_dict['to_be'],
        alpha=.4,
    )

    ax[kpi_idx].grid()

fig.tight_layout()
plot_dir = 'plot'
fname = os.path.join(plot_dir, "mem_" + str(enb_id)+ "_" + str(cell_id) +".png" )
fig.savefig(fname, format='png')
plt.close(fig)
print("Plots are saved in {}".format(fname))
# fig.show()
