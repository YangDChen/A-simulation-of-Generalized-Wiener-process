# Generalized Wiener Process Simulation

以Python模擬並視覺化零漂移的一般化 Wiener process，展示標準常態衝擊如何經由時間尺度調整、累積與線性轉換，形成隨機過程的單一路徑，以及路徑上的離散觀察值。

核心程式為 [`GWP_simulation.py`](GWP_simulation.py)，不需要額外下載外部資料集。

## 專案目的與功能

- 由獨立標準常態樣本建立 Brownian increments。
- 累積增量，得到從零開始的標準 Brownian / Wiener process。
- 透過初始值與波動係數，建立 $S_t=S_0+\sigma W_t$。
- 繪製一條模擬路徑，標示指定時間的觀察值、垂直輔助線及數學符號。

目前程式採固定參數、單一路徑模擬，未實作非零漂移、多路徑統計、參數估計或命令列參數介面。

## 數學角度

### 標準 Brownian / Wiener process

標準Wiener process $W_t$ 滿足Initial value $W_0 = 0$、路徑幾乎必然連續，且不重疊時間區間的增量互相獨立。對於 $0\le s<t$：

$$
W_t-W_s\sim\mathcal{N}(0,t-s).
$$

$\mathcal{N}(m,v)$ 的第二個參數為變異數。因此：

$$
\mathbb{E}[W_t]=0,\qquad
Var(W_t)=t,\qquad
Cov(W_s,W_t)=\min(s,t).
$$

### 模型

常數係數的一般化Wiener process可寫為：

$$
dS_t=\mu\,dt+\sigma\,dW_t,
\qquad S_t=S_0+\mu t+\sigma W_t.
$$

**目前程式未加入漂移項，相當於設定 $\mu=0$：**

$$
S_t=S_0+\sigma W_t,
\qquad S_t\sim\mathcal{N}(S_0,\sigma^2t).
$$

因此 $S_t-S_s\sim\mathcal{N}(0,\sigma^2(t-s))$ 非geometric Brownian motion，其值可能為負，沒有保證價格為正的機制。 $\sigma$ 是每平方根時間單位的波動度，而非百分比報酬波動度。

## 模擬邏輯

1. 將時間區間 $[0,T]$ 均分為 $N$ 段：

$$
\Delta t=T/N,\qquad t_i=i\Delta t,\quad i=0,1,\ldots,N.
$$

3. 產生 $N$ 個獨立標準常態衝擊，並縮放為Brownian increments：

$$
\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,1),\qquad
\Delta W_i=\sqrt{\Delta t}\,\varepsilon_i\sim\mathcal{N}(0,\Delta t),\quad i=1,\ldots,N.
$$

   必須乘上 $\sqrt{\Delta t}$，因為變異數會隨乘數的平方縮放。

4. 累積增量並加入初始值：

$$
W_{t_0}=0,\qquad W_{t_k}=\sum_{i=1}^{k}\Delta W_i,
\qquad S_{t_k}=S_0+\sigma W_{t_k}.
$$

5. 對每個指定觀察時間，以 `np.argmin(np.abs(t - time))` 尋找最近的tick位置，再擷取 `S` 的對應數值。
6. 用Matplotlib畫路徑與觀察點。

核心運算對應如下：

```python
e = np.random.normal(loc=0, scale=1, size=N)
dW = np.sqrt(dt) * e
W = np.concatenate([[0], np.cumsum(dW)])
S = S_t0 + sigma * W
```

| 變數 | 預設形狀 | 意義 |
| --- | --- | --- |
| `e`, `dW` | `(3000,)` | 每個時間區間的衝擊與增量 |
| `t`, `W`, `S` | `(3001,)` | 包含起點的時間tick與過程value |
| `observation_values` | `(6,)` | 六個指定時間的觀察值 |

## 參數說明

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `SEED` | `5566` | NumPy 隨機種子 |
| `T` | `6` | 模擬終點；起點為 `0` |
| `N` | `3000` | 時間區間數，須為正整數 |
| `dt` | `T / N = 0.002` | 時間步長，由前兩項計算 |
| `S_t0` | `2.0` | 初始值 $S_0$ |
| `sigma` | `0.30` | 波動度，這裡我是自行假設。可以參考過去歷史資料取得波動度，或使用GARCH model來取得合理波動度|
| `observation_times` | `[1, 2, 3, 4, 5, 6]` | 圖上的觀察時間 |

`T` 須大於零。時間單位由使用情境決定；程式沒有指定年、日或秒，解讀 `sigma` 時須使用一致的時間單位。

預設觀察時間對應tick的index `[500, 1000, 1500, 2000, 2500, 3000]`。圖中的 $t_1,t_2,\ldots,t_n$ 是觀察點的示意標籤；例如圖上的 $t_1$ 代表時間 `1`，不是tick的第一步 `0.002`。
