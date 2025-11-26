# This file is auto-generated. Do not edit it manually.
import numpy as np
import torch


class GeneratedGammaCalculator:
    def __init__(self):
        self.feature_names = ['temperature_C', 'wind_speed_ms', 'illuminance_lux', 'precipitation_mm', 'elec_vmin', 'elec_vmax', 'elec_vavg', 'elec_line_loading_max', 'elec_line_loading_top1', 'elec_line_loading_top2', 'elec_line_loading_top3', 'elec_vmax_nonslack', 'elec_line_i_ka_max', 'elec_p_loss_mw', 'elec_soc_1', 'elec_soc_2', 'elec_soc_3', 'elec_soc_4', 'elec_soc_5', 'elec_any_violate', 'elec_violate_v', 'elec_violate_line']
        self.gamma_signals = ['yhat_ess1_up', 'yhat_ess1_down', 'yhat_ess2_up', 'yhat_ess2_down', 'yhat_ess3_up', 'yhat_ess3_down', 'yhat_ess4_up', 'yhat_ess4_down', 'yhat_ess5_up', 'yhat_ess5_down', 'yhat_sop1_fwd', 'yhat_sop1_rev', 'yhat_sop2_fwd', 'yhat_sop2_rev', 'yhat_pv1', 'yhat_pv2', 'yhat_pv3', 'yhat_pv4', 'yhat_pv5', 'yhat_pv6', 'yhat_wt1', 'yhat_wt2', 'yhat_wt3', 'yhat_wt4', 'yhat_wt5', 'yhat_wt6']

    def compute(self, state):
        # Unpack state variables to match PySR's x_i notation
        x0 = state['temperature_C']
        x1 = state['wind_speed_ms']
        x2 = state['illuminance_lux']
        x3 = state['precipitation_mm']
        x4 = state['elec_vmin']
        x5 = state['elec_vmax']
        x6 = state['elec_vavg']
        x7 = state['elec_line_loading_max']
        x8 = state['elec_line_loading_top1']
        x9 = state['elec_line_loading_top2']
        x10 = state['elec_line_loading_top3']
        x11 = state['elec_vmax_nonslack']
        x12 = state['elec_line_i_ka_max']
        x13 = state['elec_p_loss_mw']
        x14 = state['elec_soc_1']
        x15 = state['elec_soc_2']
        x16 = state['elec_soc_3']
        x17 = state['elec_soc_4']
        x18 = state['elec_soc_5']
        x19 = state['elec_any_violate']
        x20 = state['elec_violate_v']
        x21 = state['elec_violate_line']

        # Calculate each gamma component
        gamma_0 = x20  # Formula for yhat_ess1_up
        gamma_1 = x19 * np.cos(((x21 - (np.sin(x1) * np.sin(x1))) * -0.00081227196) / np.sin(0.43504593 * x0))  # Formula for yhat_ess1_down
        gamma_2 = (((x19 - 0.9745499) * np.cos(x13 * x9)) * (x12 * np.sin(-0.20329778 * x9))) + x20  # Formula for yhat_ess2_up
        gamma_3 = x19 * np.cos(x19 + (np.exp(np.cos(np.log(np.cos((np.log(x13) + x12) * x19) * 0.47853702) + -1.6723061)) + -1.5096939))  # Formula for yhat_ess2_down
        gamma_4 = np.sin((4.2981977 / np.sin(-14.550615 * x4)) * np.sin(x20 + -5.471902))  # Formula for yhat_ess3_up
        gamma_5 = np.cos((np.sin(x8 * x13) * np.sin(2.6682842 - (x18 * x4))) * ((x20 - np.cos(x17 * (x13 * 1.8024681))) - -0.29180428)) * x20  # Formula for yhat_ess3_down
        gamma_6 = x19 + (np.exp(np.sin(x10 + 0.733823) * np.exp(np.cos(x2 + x19))) / x0)  # Formula for yhat_ess4_up
        gamma_7 = x11 * np.sin(x20 / ((np.cos(x2 / 0.7342755) / x0) - np.sin(np.sin(x8 / ((x16 - x8) + -0.37565067)))))  # Formula for yhat_ess4_down
        gamma_8 = np.exp(-5.3776984 - ((np.cos(x9 * 0.6933178) + (x19 - np.sin(np.cos(x9 * 0.27743956)))) * 2.1182005)) + x20  # Formula for yhat_ess5_up
        gamma_9 = x19 * np.cos(x14 * (-0.40549272 / ((x16 / x4) - (x14 / np.sin(np.sin(0.79703146 / (np.sin(x16 / x4) - 0.8874391)))))))  # Formula for yhat_ess5_down
        gamma_10 = np.sin(x20 + np.sin((x7 * np.log(x5 + x19)) * 0.09440813))  # Formula for yhat_sop1_fwd
        gamma_11 = np.cos(0.66600674 - np.cos(np.log(np.log(x7)) * (-15.497216 / x4)))  # Formula for yhat_sop1_rev
        gamma_12 = x20 * np.sin(np.cos(np.sin(((0.5383529 - x13) * x16) - x13) - (0.73255324 - x13)) / 0.5450916)  # Formula for yhat_sop2_fwd
        gamma_13 = np.cos(np.sin((x17 / x6) - 1.6084003))  # Formula for yhat_sop2_rev
        gamma_14 = x19  # Formula for yhat_pv1
        gamma_15 = x19  # Formula for yhat_pv2
        gamma_16 = x20  # Formula for yhat_pv3
        gamma_17 = x19  # Formula for yhat_pv4
        gamma_18 = x20  # Formula for yhat_pv5
        gamma_19 = x20 * np.cos(x13 - x21)  # Formula for yhat_pv6
        gamma_20 = x19  # Formula for yhat_wt1
        gamma_21 = x20  # Formula for yhat_wt2
        gamma_22 = x19  # Formula for yhat_wt3
        gamma_23 = x20  # Formula for yhat_wt4
        gamma_24 = x19  # Formula for yhat_wt5
        gamma_25 = x20  # Formula for yhat_wt6

        # Combine components into a vector
        gamma_values = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, gamma_5, gamma_6, gamma_7, gamma_8, gamma_9, gamma_10, gamma_11, gamma_12, gamma_13, gamma_14, gamma_15, gamma_16, gamma_17, gamma_18, gamma_19, gamma_20, gamma_21, gamma_22, gamma_23, gamma_24, gamma_25]

        # Convert to a PyTorch tensor with a batch dimension
        return torch.tensor(gamma_values, dtype=torch.float32).unsqueeze(0)