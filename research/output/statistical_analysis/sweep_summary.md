| hyperparameter | method | metric | n_files | default_value | metric_at_default | best_value | metric_at_best | gain_over_default | median_time_ms_at_default | median_time_ms_at_best |
|---|---|---|---|---|---|---|---|---|---|---|
| ar_order | LPC | SNR (dB) | 1 | 256 | 19.3 | 512 | 20.32 | 1.025 | 125.8 | 223.5 |
| ar_order | LPC | MR-STFT dist. | 1 | 256 | 0.1648 | 1024 | 0.1538 | 0.011 | 125.8 | 436.5 |
| ar_order | LPC | PLCMOS | 1 | 256 | 2.994 | 1024 | 3.059 | 0.06487 | 125.8 | 436.5 |
| ar_order | LPC | ViSQOL MOS-LQO | 1 | 256 | 4.714 | 1024 | 4.717 | 0.002738 | 125.8 | 436.5 |
| ar_order | NN | SNR (dB) | 1 | 256 | 19.12 | 1536 | 19.38 | 0.2597 | 2469 | 3227 |
| ar_order | NN | MR-STFT dist. | 1 | 256 | 0.1755 | 1024 | 0.1725 | 0.003031 | 2469 | 2927 |
| ar_order | NN | PLCMOS | 1 | 256 | 3.027 | 1536 | 3.08 | 0.05215 | 2469 | 3227 |
| ar_order | NN | ViSQOL MOS-LQO | 1 | 256 | 4.71 | 128 | 4.71 | 0.0007882 | 2469 | 2218 |
| context_length_packets | LPC | SNR (dB) | 1 | 8 | 19.3 | 8 | 19.3 | 0 | 142 | 142 |
| context_length_packets | LPC | MR-STFT dist. | 1 | 8 | 0.1648 | 8 | 0.1648 | 0 | 142 | 142 |
| context_length_packets | LPC | PLCMOS | 1 | 8 | 2.991 | 8 | 2.991 | 0 | 142 | 142 |
| context_length_packets | LPC | ViSQOL MOS-LQO | 1 | 8 | 4.714 | 16 | 4.716 | 0.001491 | 142 | 155.7 |
| context_length_packets | NN | SNR (dB) | 1 | 8 | 19.12 | 8 | 19.12 | 0 | 2618 | 2618 |
| context_length_packets | NN | MR-STFT dist. | 1 | 8 | 0.1755 | 8 | 0.1755 | 0 | 2618 | 2618 |
| context_length_packets | NN | PLCMOS | 1 | 8 | 3.006 | 8 | 3.006 | 0 | 2618 | 2618 |
| context_length_packets | NN | ViSQOL MOS-LQO | 1 | 8 | 4.71 | 8 | 4.71 | 0 | 2618 | 2618 |
| diagonal_load | LPC | SNR (dB) | 1 | 0.001 | 19.3 | 1e-05 | 19.3 | 0.005188 | 157 | 152.9 |
| diagonal_load | LPC | MR-STFT dist. | 1 | 0.001 | 0.1648 | 1e-06 | 0.1641 | 0.0006889 | 157 | 139.4 |
| diagonal_load | LPC | PLCMOS | 1 | 0.001 | 2.975 | 1e-05 | 2.994 | 0.01878 | 157 | 152.9 |
| diagonal_load | LPC | ViSQOL MOS-LQO | 1 | 0.001 | 4.714 | 1e-06 | 4.714 | 0.0001263 | 157 | 139.4 |
| diagonal_load | NN | SNR (dB) | 1 | 0.001 | 19.12 | 1e-06 | 19.13 | 0.008492 | 3053 | 4834 |
| diagonal_load | NN | MR-STFT dist. | 1 | 0.001 | 0.1755 | 1e-06 | 0.1748 | 0.0007052 | 3053 | 4834 |
| diagonal_load | NN | PLCMOS | 1 | 0.001 | 3.019 | 1e-05 | 3.031 | 0.01196 | 3053 | 2811 |
| diagonal_load | NN | ViSQOL MOS-LQO | 1 | 0.001 | 4.71 | 1e-06 | 4.71 | 0.000686 | 3053 | 4834 |
| extra_dim | LPC | SNR (dB) | 1 | 256 | 19.3 | 16 | 19.68 | 0.3855 | 159.3 | 110.8 |
| extra_dim | LPC | MR-STFT dist. | 1 | 256 | 0.1648 | 64 | 0.1523 | 0.01253 | 159.3 | 137.1 |
| extra_dim | LPC | PLCMOS | 1 | 256 | 2.967 | 512 | 3.004 | 0.03699 | 159.3 | 172.1 |
| extra_dim | LPC | ViSQOL MOS-LQO | 1 | 256 | 4.714 | 32 | 4.717 | 0.002485 | 159.3 | 129.2 |
| extra_dim | NN | SNR (dB) | 1 | 256 | 19.12 | 16 | 19.43 | 0.312 | 3071 | 4192 |
| extra_dim | NN | MR-STFT dist. | 1 | 256 | 0.1755 | 128 | 0.1733 | 0.002208 | 3071 | 2968 |
| extra_dim | NN | PLCMOS | 1 | 256 | 3.058 | 256 | 3.058 | 0 | 3071 | 3071 |
| extra_dim | NN | ViSQOL MOS-LQO | 1 | 256 | 4.71 | 192 | 4.71 | 0.0003969 | 3071 | 3125 |
