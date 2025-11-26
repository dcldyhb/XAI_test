# PySR Gamma 信号公式汇总

## 特征索引映射表

公式中的 $x_i$ 符号与以下实际特征相对应：

| 符号 ($x_i$) | 特征变量名 |
| :--- | :--- |
| $x_{0}$ | `temperature_C` |
| $x_{1}$ | `wind_speed_ms` |
| $x_{2}$ | `illuminance_lux` |
| $x_{3}$ | `precipitation_mm` |
| $x_{4}$ | `elec_vmin` |
| $x_{5}$ | `elec_vmax` |
| $x_{6}$ | `elec_vavg` |
| $x_{7}$ | `elec_line_loading_max` |
| $x_{8}$ | `elec_line_loading_top1` |
| $x_{9}$ | `elec_line_loading_top2` |
| $x_{10}$ | `elec_line_loading_top3` |
| $x_{11}$ | `elec_vmax_nonslack` |
| $x_{12}$ | `elec_line_i_ka_max` |
| $x_{13}$ | `elec_p_loss_mw` |
| $x_{14}$ | `elec_soc_1` |
| $x_{15}$ | `elec_soc_2` |
| $x_{16}$ | `elec_soc_3` |
| $x_{17}$ | `elec_soc_4` |
| $x_{18}$ | `elec_soc_5` |
| $x_{19}$ | `elec_any_violate` |
| $x_{20}$ | `elec_violate_v` |
| $x_{21}$ | `elec_violate_line` |


---

### Gamma 信号: `yhat_ess1_up`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_ess1_down`

**最佳公式:**

$$
x_{19} \cos{\left(\left(x_{21} - \sin{\left(x_{1} \right)} \sin{\left(x_{1} \right)}\right) \left(-0.0008123\right) \frac{1}{\sin{\left(0.4350 x_{0} \right)}} \right)}
$$

---

### Gamma 信号: `yhat_ess2_up`

**最佳公式:**

$$
x_{12} \left(x_{19} - 0.9745\right) \sin{\left(- 0.2033 x_{9} \right)} \cos{\left(x_{13} x_{9} \right)} + x_{20}
$$

---

### Gamma 信号: `yhat_ess2_down`

**最佳公式:**

$$
x_{19} \cos{\left(x_{19} + e^{\cos{\left(\log{\left(0.4785 \cos{\left(x_{19} \left(x_{12} + \log{\left(x_{13} \right)}\right) \right)} \right)} - 1.672 \right)}} - 1.510 \right)}
$$

---

### Gamma 信号: `yhat_ess3_up`

**最佳公式:**

$$
\sin{\left(\frac{4.298 \sin{\left(x_{20} - 5.472 \right)}}{\sin{\left(- 14.55 x_{4} \right)}} \right)}
$$

---

### Gamma 信号: `yhat_ess3_down`

**最佳公式:**

$$
x_{20} \cos{\left(\left(x_{20} - \cos{\left(x_{17} x_{13} \cdot 1.802 \right)} - -0.2918\right) \sin{\left(x_{13} x_{8} \right)} \sin{\left(2.668 - x_{18} x_{4} \right)} \right)}
$$

---

### Gamma 信号: `yhat_ess4_up`

**最佳公式:**

$$
x_{19} + \frac{e^{e^{\cos{\left(x_{19} + x_{2} \right)}} \sin{\left(x_{10} + 0.7338 \right)}}}{x_{0}}
$$

---

### Gamma 信号: `yhat_ess4_down`

**最佳公式:**

$$
x_{11} \sin{\left(\frac{x_{20}}{- \sin{\left(\sin{\left(\frac{x_{8}}{x_{16} - x_{8} - 0.3757} \right)} \right)} + \frac{\cos{\left(\frac{x_{2}}{0.7343} \right)}}{x_{0}}} \right)}
$$

---

### Gamma 信号: `yhat_ess5_up`

**最佳公式:**

$$
x_{20} + e^{- 2.118 \left(x_{19} - \sin{\left(\cos{\left(x_{9} \cdot 0.2774 \right)} \right)} + \cos{\left(x_{9} \cdot 0.6933 \right)}\right) - 5.378}
$$

---

### Gamma 信号: `yhat_ess5_down`

**最佳公式:**

$$
x_{19} \cos{\left(x_{14} \left(-0.4055\right) \frac{1}{- \frac{x_{14}}{\sin{\left(\sin{\left(\frac{0.7970}{\sin{\left(\frac{x_{16}}{x_{4}} \right)} - 0.8874} \right)} \right)}} + \frac{x_{16}}{x_{4}}} \right)}
$$

---

### Gamma 信号: `yhat_sop1_fwd`

**最佳公式:**

$$
\sin{\left(x_{20} + \sin{\left(0.09441 x_{7} \log{\left(x_{19} + x_{5} \right)} \right)} \right)}
$$

---

### Gamma 信号: `yhat_sop1_rev`

**最佳公式:**

$$
\cos{\left(\cos{\left(\frac{15.50 \log{\left(\log{\left(x_{7} \right)} \right)}}{x_{4}} \right)} - 0.6660 \right)}
$$

---

### Gamma 信号: `yhat_sop2_fwd`

**最佳公式:**

$$
x_{20} \sin{\left(\frac{\cos{\left(- (0.7326 - x_{13}) + \sin{\left(- x_{13} + x_{16} \left(0.5384 - x_{13}\right) \right)} \right)}}{0.5451} \right)}
$$

---

### Gamma 信号: `yhat_sop2_rev`

**最佳公式:**

$$
\cos{\left(\sin{\left(\frac{x_{17}}{x_{6}} - 1.608 \right)} \right)}
$$

---

### Gamma 信号: `yhat_pv1`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_pv2`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_pv3`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_pv4`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_pv5`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_pv6`

**最佳公式:**

$$
x_{20} \cos{\left(x_{13} - x_{21} \right)}
$$

---

### Gamma 信号: `yhat_wt1`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt2`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_wt3`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt4`

**最佳公式:**

$$
x_{20}
$$

---

### Gamma 信号: `yhat_wt5`

**最佳公式:**

$$
x_{19}
$$

---

### Gamma 信号: `yhat_wt6`

**最佳公式:**

$$
x_{20}
$$
