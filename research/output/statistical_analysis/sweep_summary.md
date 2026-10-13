| hyperparameter | method | metric | n_files | default_value | metric_at_default | best_value | metric_at_best | gain_over_default | median_time_ms_at_default | median_time_ms_at_best |
|---|---|---|---|---|---|---|---|---|---|---|
| ar_order | This LPC | SNR (dB) | 30 | 256 | 15.25 | 768 | 16.66 | 1.416 | 149.8 | 393.4 |
| ar_order | This LPC | MR-STFT dist. | 30 | 256 | 0.4295 | 2048 | 0.3596 | 0.06991 | 149.8 | 1769 |
| ar_order | This LPC | PLCMOS | 30 | 256 | 2.508 | 1536 | 2.638 | 0.1301 | 149.8 | 983.6 |
| ar_order | This LPC | ViSQOL MOS-LQO | 30 | 256 | 4.476 | 1024 | 4.504 | 0.02809 | 149.8 | 712.1 |
| ar_order | This NN | SNR (dB) | 30 | 256 | 15.07 | 512 | 15.5 | 0.4346 | 4074 | 3939 |
| ar_order | This NN | MR-STFT dist. | 30 | 256 | 0.335 | 768 | 0.3282 | 0.006831 | 4074 | 4442 |
| ar_order | This NN | PLCMOS | 30 | 256 | 2.548 | 768 | 2.618 | 0.06953 | 4074 | 4442 |
| ar_order | This NN | ViSQOL MOS-LQO | 30 | 256 | 4.485 | 512 | 4.5 | 0.01429 | 4074 | 3939 |
| burg_order_pb | This Sparse AR | SNR (dB) | 30 | 8 | 11.38 | 128 | 14.42 | 3.041 | 175.5 | 271.3 |
| burg_order_pb | This Sparse AR | MR-STFT dist. | 30 | 8 | 0.5822 | 128 | 0.4326 | 0.1496 | 175.5 | 271.3 |
| burg_order_pb | This Sparse AR | PLCMOS | 30 | 8 | 2.088 | 128 | 2.452 | 0.364 | 175.5 | 271.3 |
| burg_order_pb | This Sparse AR | ViSQOL MOS-LQO | 30 | 8 | 4.436 | 128 | 4.474 | 0.03861 | 175.5 | 271.3 |
| context_length_packets | This LPC | SNR (dB) | 30 | 8 | 15.25 | 8 | 15.25 | 0 | 173.7 | 173.7 |
| context_length_packets | This LPC | MR-STFT dist. | 30 | 8 | 0.4295 | 16 | 0.4278 | 0.00169 | 173.7 | 192.3 |
| context_length_packets | This LPC | PLCMOS | 30 | 8 | 2.505 | 16 | 2.511 | 0.00651 | 173.7 | 192.3 |
| context_length_packets | This LPC | ViSQOL MOS-LQO | 30 | 8 | 4.476 | 32 | 4.477 | 0.001236 | 173.7 | 189.6 |
| context_length_packets | This NN | SNR (dB) | 30 | 8 | 15.07 | 8 | 15.07 | 0 | 3732 | 3732 |
| context_length_packets | This NN | MR-STFT dist. | 30 | 8 | 0.335 | 8 | 0.335 | 0 | 3732 | 3732 |
| context_length_packets | This NN | PLCMOS | 30 | 8 | 2.546 | 8 | 2.546 | 0 | 3732 | 3732 |
| context_length_packets | This NN | ViSQOL MOS-LQO | 30 | 8 | 4.485 | 8 | 4.485 | 0 | 3732 | 3732 |
| diagonal_load | This LPC | SNR (dB) | 30 | 0.001 | 15.25 | 0.0001 | 15.25 | 0.001882 | 148.4 | 140.8 |
| diagonal_load | This LPC | MR-STFT dist. | 30 | 0.001 | 0.4295 | 1e-06 | 0.4172 | 0.01236 | 148.4 | 151.3 |
| diagonal_load | This LPC | PLCMOS | 30 | 0.001 | 2.497 | 1e-05 | 2.505 | 0.008476 | 148.4 | 153.6 |
| diagonal_load | This LPC | ViSQOL MOS-LQO | 30 | 0.001 | 4.476 | 0.0001 | 4.477 | 0.001164 | 148.4 | 140.8 |
| diagonal_load | This NN | SNR (dB) | 30 | 0.001 | 15.07 | 1e-05 | 15.07 | 0.003858 | 3715 | 3318 |
| diagonal_load | This NN | MR-STFT dist. | 30 | 0.001 | 0.335 | 1e-05 | 0.3342 | 0.000773 | 3715 | 3318 |
| diagonal_load | This NN | PLCMOS | 30 | 0.001 | 2.547 | 0.001 | 2.547 | 0 | 3715 | 3715 |
| diagonal_load | This NN | ViSQOL MOS-LQO | 30 | 0.001 | 4.485 | 1e-06 | 4.487 | 0.00108 | 3715 | 3121 |
| extra_dim | This LPC | SNR (dB) | 30 | 256 | 15.25 | 16 | 16.06 | 0.8088 | 160 | 144.1 |
| extra_dim | This LPC | MR-STFT dist. | 30 | 256 | 0.4295 | 32 | 0.3801 | 0.04946 | 160 | 102.1 |
| extra_dim | This LPC | PLCMOS | 30 | 256 | 2.497 | 512 | 2.533 | 0.03634 | 160 | 171.7 |
| extra_dim | This LPC | ViSQOL MOS-LQO | 30 | 256 | 4.476 | 32 | 4.524 | 0.04789 | 160 | 102.1 |
| extra_dim | This NN | SNR (dB) | 30 | 256 | 15.07 | 16 | 15.9 | 0.8339 | 3830 | 3137 |
| extra_dim | This NN | MR-STFT dist. | 30 | 256 | 0.335 | 64 | 0.3141 | 0.02092 | 3830 | 3274 |
| extra_dim | This NN | PLCMOS | 30 | 256 | 2.543 | 384 | 2.563 | 0.02009 | 3830 | 3505 |
| extra_dim | This NN | ViSQOL MOS-LQO | 30 | 256 | 4.485 | 64 | 4.503 | 0.01717 | 3830 | 3274 |
| extra_dim | This Sparse AR | SNR (dB) | 30 | 256 | 11.38 | 128 | 11.76 | 0.3804 | 157.3 | 2157 |
| extra_dim | This Sparse AR | MR-STFT dist. | 30 | 256 | 0.5822 | 192 | 0.568 | 0.01417 | 157.3 | 204.9 |
| extra_dim | This Sparse AR | PLCMOS | 30 | 256 | 2.111 | 256 | 2.111 | 0 | 157.3 | 157.3 |
| extra_dim | This Sparse AR | ViSQOL MOS-LQO | 30 | 256 | 4.436 | 128 | 4.554 | 0.118 | 157.3 | 2157 |
| lambda_reg | This Sparse AR | SNR (dB) | 30 | 0.01 | 11.38 | 0.0001 | 11.38 | 0.001595 | 166 | 241 |
| lambda_reg | This Sparse AR | MR-STFT dist. | 30 | 0.01 | 0.5822 | 0.0001 | 0.5819 | 0.0002352 | 166 | 241 |
| lambda_reg | This Sparse AR | PLCMOS | 30 | 0.01 | 2.089 | 0.001 | 2.104 | 0.01507 | 166 | 192.9 |
| lambda_reg | This Sparse AR | ViSQOL MOS-LQO | 30 | 0.01 | 4.436 | 0.0001 | 4.436 | 6.38e-05 | 166 | 241 |
| max_k_lags | This Sparse AR | SNR (dB) | 30 | 3 | 11.38 | 4 | 11.38 | 0.0001009 | 179.3 | 172 |
| max_k_lags | This Sparse AR | MR-STFT dist. | 30 | 3 | 0.5822 | 4 | 0.5821 | 4.732e-06 | 179.3 | 172 |
| max_k_lags | This Sparse AR | PLCMOS | 30 | 3 | 2.089 | 6 | 2.11 | 0.02109 | 179.3 | 177.6 |
| max_k_lags | This Sparse AR | ViSQOL MOS-LQO | 30 | 3 | 4.436 | 4 | 4.436 | 9.548e-07 | 179.3 | 172 |
| max_lag_l | This Sparse AR | SNR (dB) | 30 | 1024 | 11.38 | 1024 | 11.38 | 0 | 150.1 | 150.1 |
| max_lag_l | This Sparse AR | MR-STFT dist. | 30 | 1024 | 0.5822 | 1024 | 0.5822 | 0 | 150.1 | 150.1 |
| max_lag_l | This Sparse AR | PLCMOS | 30 | 1024 | 2.105 | 1024 | 2.105 | 0 | 150.1 | 150.1 |
| max_lag_l | This Sparse AR | ViSQOL MOS-LQO | 30 | 1024 | 4.436 | 1024 | 4.436 | 0 | 150.1 | 150.1 |
