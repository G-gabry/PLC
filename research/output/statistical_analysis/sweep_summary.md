| hyperparameter | method | metric | n_files | default_value | metric_at_default | best_value | metric_at_best | gain_over_default | median_time_ms_at_default | median_time_ms_at_best |
|---|---|---|---|---|---|---|---|---|---|---|
| ar_order | This LPC | SNR (dB) | 300 | 256 | 15.5 | 1024 | 16.65 | 1.152 | 156.2 | 602.4 |
| ar_order | This LPC | MR-STFT dist. | 300 | 256 | 0.4702 | 2048 | 0.3957 | 0.07444 | 156.2 | 1499 |
| ar_order | This LPC | PLCMOS | 300 | 256 | 2.396 | 1024 | 2.501 | 0.1056 | 156.2 | 602.4 |
| ar_order | This LPC | ViSQOL MOS-LQO | 300 | 256 | 4.519 | 1024 | 4.539 | 0.02019 | 156.2 | 602.4 |
| ar_order | This NN | SNR (dB) | 300 | 256 | 15.3 | 768 | 15.47 | 0.1703 | 3980 | 4214 |
| ar_order | This NN | MR-STFT dist. | 300 | 256 | 0.3595 | 512 | 0.3585 | 0.001054 | 3980 | 3976 |
| ar_order | This NN | PLCMOS | 300 | 256 | 2.452 | 768 | 2.51 | 0.05851 | 3980 | 4214 |
| ar_order | This NN | ViSQOL MOS-LQO | 300 | 256 | 4.505 | 384 | 4.513 | 0.008205 | 3980 | 3954 |
| burg_order_pb | This Sparse AR | SNR (dB) | 300 | 8 | 11.74 | 128 | 14.85 | 3.111 | 132.9 | 253.4 |
| burg_order_pb | This Sparse AR | MR-STFT dist. | 300 | 8 | 0.6138 | 128 | 0.4621 | 0.1517 | 132.9 | 253.4 |
| burg_order_pb | This Sparse AR | PLCMOS | 300 | 8 | 2.048 | 128 | 2.351 | 0.3036 | 132.9 | 253.4 |
| burg_order_pb | This Sparse AR | ViSQOL MOS-LQO | 300 | 8 | 4.476 | 128 | 4.527 | 0.05105 | 132.9 | 253.4 |
| context_length_packets | This LPC | SNR (dB) | 300 | 8 | 15.5 | 16 | 15.5 | 0.006832 | 158.6 | 178.3 |
| context_length_packets | This LPC | MR-STFT dist. | 300 | 8 | 0.4702 | 16 | 0.4655 | 0.004656 | 158.6 | 178.3 |
| context_length_packets | This LPC | PLCMOS | 300 | 8 | 2.396 | 16 | 2.42 | 0.02422 | 158.6 | 178.3 |
| context_length_packets | This LPC | ViSQOL MOS-LQO | 300 | 8 | 4.519 | 32 | 4.523 | 0.004101 | 158.6 | 197.6 |
| context_length_packets | This NN | SNR (dB) | 300 | 8 | 15.3 | 8 | 15.3 | 0 | 3968 | 3968 |
| context_length_packets | This NN | MR-STFT dist. | 300 | 8 | 0.3595 | 8 | 0.3595 | 0 | 3968 | 3968 |
| context_length_packets | This NN | PLCMOS | 300 | 8 | 2.452 | 8 | 2.452 | 0 | 3968 | 3968 |
| context_length_packets | This NN | ViSQOL MOS-LQO | 300 | 8 | 4.505 | 8 | 4.505 | 0 | 3968 | 3968 |
| context_length_packets | This Sparse AR | SNR (dB) | 300 | 8 | 7.854 | 32 | 7.878 | 0.02384 | 1980 | 1929 |
| context_length_packets | This Sparse AR | MR-STFT dist. | 300 | 8 | 1.399 | 256 | 1.271 | 0.1278 | 1980 | 3072 |
| context_length_packets | This Sparse AR | PLCMOS | 300 | 8 | 1.707 | 4 | 1.834 | 0.1271 | 1980 | 2073 |
| context_length_packets | This Sparse AR | ViSQOL MOS-LQO | 300 | 8 | 4.166 | 32 | 4.17 | 0.003409 | 1980 | 1929 |
| diagonal_load | This LPC | SNR (dB) | 300 | 0.001 | 15.5 | 0.0001 | 15.5 | 0.001326 | 150.6 | 140.9 |
| diagonal_load | This LPC | MR-STFT dist. | 300 | 0.001 | 0.4702 | 1e-06 | 0.4562 | 0.01398 | 150.6 | 129.7 |
| diagonal_load | This LPC | PLCMOS | 300 | 0.001 | 2.396 | 0.0001 | 2.399 | 0.003221 | 150.6 | 140.9 |
| diagonal_load | This LPC | ViSQOL MOS-LQO | 300 | 0.001 | 4.519 | 1e-05 | 4.52 | 0.001393 | 150.6 | 140 |
| diagonal_load | This NN | SNR (dB) | 300 | 0.001 | 15.3 | 0.0001 | 15.3 | 0.00227 | 3512 | 3483 |
| diagonal_load | This NN | MR-STFT dist. | 300 | 0.001 | 0.3595 | 1e-05 | 0.3588 | 0.0006664 | 3512 | 3442 |
| diagonal_load | This NN | PLCMOS | 300 | 0.001 | 2.451 | 0.01 | 2.455 | 0.003572 | 3512 | 3530 |
| diagonal_load | This NN | ViSQOL MOS-LQO | 300 | 0.001 | 4.505 | 1e-05 | 4.506 | 0.00115 | 3512 | 3442 |
| extra_dim | This LPC | SNR (dB) | 300 | 256 | 15.5 | 16 | 16.37 | 0.8676 | 111.3 | 121 |
| extra_dim | This LPC | MR-STFT dist. | 300 | 256 | 0.4702 | 32 | 0.4193 | 0.05092 | 111.3 | 109.6 |
| extra_dim | This LPC | PLCMOS | 300 | 256 | 2.392 | 512 | 2.414 | 0.02106 | 111.3 | 127 |
| extra_dim | This LPC | ViSQOL MOS-LQO | 300 | 256 | 4.519 | 32 | 4.556 | 0.03668 | 111.3 | 109.6 |
| extra_dim | This NN | SNR (dB) | 300 | 256 | 15.3 | 16 | 16.13 | 0.8299 | 2762 | 3292 |
| extra_dim | This NN | MR-STFT dist. | 300 | 256 | 0.3595 | 64 | 0.3385 | 0.02102 | 2762 | 3455 |
| extra_dim | This NN | PLCMOS | 300 | 256 | 2.451 | 384 | 2.456 | 0.005133 | 2762 | 2736 |
| extra_dim | This NN | ViSQOL MOS-LQO | 300 | 256 | 4.505 | 128 | 4.515 | 0.01073 | 2762 | 3174 |
| extra_dim | This Sparse AR | SNR (dB) | 300 | 256 | 11.74 | 128 | 12.72 | 0.9737 | 138.9 | 163.6 |
| extra_dim | This Sparse AR | MR-STFT dist. | 300 | 256 | 0.6138 | 128 | 0.5975 | 0.01632 | 138.9 | 163.6 |
| extra_dim | This Sparse AR | PLCMOS | 300 | 256 | 2.05 | 256 | 2.05 | 0 | 138.9 | 138.9 |
| extra_dim | This Sparse AR | ViSQOL MOS-LQO | 300 | 256 | 4.476 | 128 | 4.569 | 0.09335 | 138.9 | 163.6 |
| lambda_reg | This Sparse AR | SNR (dB) | 300 | 0.01 | 11.74 | 0.001 | 11.75 | 0.006105 | 136.6 | 136.7 |
| lambda_reg | This Sparse AR | MR-STFT dist. | 300 | 0.01 | 0.6138 | 0.0001 | 0.6132 | 0.0005569 | 136.6 | 152.6 |
| lambda_reg | This Sparse AR | PLCMOS | 300 | 0.01 | 2.047 | 0.001 | 2.051 | 0.004317 | 136.6 | 136.7 |
| lambda_reg | This Sparse AR | ViSQOL MOS-LQO | 300 | 0.01 | 4.476 | 0.001 | 4.476 | 0.0001352 | 136.6 | 136.7 |
| max_k_lags | This Sparse AR | SNR (dB) | 300 | 3 | 11.74 | 3 | 11.74 | 0 | 138.5 | 138.5 |
| max_k_lags | This Sparse AR | MR-STFT dist. | 300 | 3 | 0.6138 | 4 | 0.6137 | 5.555e-05 | 138.5 | 138.5 |
| max_k_lags | This Sparse AR | PLCMOS | 300 | 3 | 2.047 | 2 | 2.05 | 0.003122 | 138.5 | 133.2 |
| max_k_lags | This Sparse AR | ViSQOL MOS-LQO | 300 | 3 | 4.476 | 4 | 4.476 | 2.364e-05 | 138.5 | 138.5 |
| max_lag_l | This Sparse AR | SNR (dB) | 300 | 1024 | 11.74 | 2048 | 11.74 | 0.0007622 | 129.1 | 220.7 |
| max_lag_l | This Sparse AR | MR-STFT dist. | 300 | 1024 | 0.6138 | 2048 | 0.6135 | 0.0003342 | 129.1 | 220.7 |
| max_lag_l | This Sparse AR | PLCMOS | 300 | 1024 | 2.05 | 1024 | 2.05 | 0 | 129.1 | 129.1 |
| max_lag_l | This Sparse AR | ViSQOL MOS-LQO | 300 | 1024 | 4.476 | 2048 | 4.476 | 2.445e-05 | 129.1 | 220.7 |
