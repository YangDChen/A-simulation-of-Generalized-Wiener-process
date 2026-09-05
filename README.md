# Generalized Wiener Process Simulation

以 Python、NumPy 與 Matplotlib 模擬並視覺化 **零漂移的一般化 Wiener process**，展示標準常態衝擊如何經由時間尺度調整、累積與線性轉換，形成隨機過程的單一路徑，以及路徑上的離散觀察值。

本專案適合用於隨機過程與財務工程的基礎學習。核心程式為 [`GWP_simulation.py`](GWP_simulation.py)，不需要外部資料集。

## 專案目的與功能

- 由獨立標準常態樣本建立 Brownian increments。
- 累積增量，得到從零開始的標準 Brownian / Wiener process。
- 透過初始值與波動係數，建立 $S_t=S_0+\sigma W_t$。
- 繪製一條模擬路徑，標示指定時間的觀察值、垂直輔助線及數學符號。

目前程式採固定參數、單一路徑模擬；未實作非零漂移、多路徑統計、參數估計或命令列參數介面。

## 數學背景

### 標準 Brownian / Wiener process

標準 Wiener process $W_t$ 滿足 $W_0=0$、路徑幾乎必然連續，且不重疊時間區間的增量互相獨立。對 $0\le s<t$：

$$
W_t-W_s\sim\mathcal{N}(0,t-s).
$$

本文 $\mathcal{N}(m,v)$ 的第二個參數為**變異數**。因此：

$$
\mathbb{E}[W_t]=0,\qquad
\operatorname{Var}(W_t)=t,\qquad
\operatorname{Cov}(W_s,W_t)=\min(s,t).
$$

獨立的是不重疊區間的增量，不是不同時間點的過程值。

### 本程式使用的模型

常數係數的一般化 Wiener process 可寫為：

$$
dS_t=\mu\,dt+\sigma\,dW_t,
\qquad S_t=S_0+\mu t+\sigma W_t.
$$

**目前程式未加入漂移項，相當於設定 $\mu=0$：**

$$
S_t=S_0+\sigma W_t,
\qquad S_t\sim\mathcal{N}(S_0,\sigma^2t).
$$

因此 $S_t-S_s\sim\mathcal{N}(0,\sigma^2(t-s))$。這是加法型模型，並非 geometric Brownian motion；其值可能為負，沒有保證價格為正的機制。$\sigma$ 是每平方根時間單位的絕對波動尺度，不應直接解讀為百分比報酬波動率。

## 模擬邏輯

1. 將時間區間 $[0,T]$ 均分為 $N$ 段：

   $$
   \Delta t=T/N,\qquad t_i=i\Delta t,\quad i=0,1,\ldots,N.
   $$

2. 產生 $N$ 個獨立標準常態衝擊，並縮放為 Brownian increments：

   $$
   \varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal{N}(0,1),\qquad
   \Delta W_i=\sqrt{\Delta t}\,\varepsilon_i\sim\mathcal{N}(0,\Delta t),\quad i=1,\ldots,N.
   $$

   必須乘上 $\sqrt{\Delta t}$，因為變異數會隨乘數的平方縮放。NumPy 的 `normal(scale=...)` 接受的是標準差。

3. 累積增量並加入初始值：

   $$
   W_{t_0}=0,\qquad W_{t_k}=\sum_{i=1}^{k}\Delta W_i,
   \qquad S_{t_k}=S_0+\sigma W_{t_k}.
   $$

4. 對每個指定觀察時間，以 `np.argmin(np.abs(t - time))` 尋找最近的網格位置，再擷取 `S` 的對應數值。
5. 使用 Matplotlib 繪製路徑與觀察點。

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
| `t`, `W`, `S` | `(3001,)` | 包含起點的時間網格與過程值 |
| `observation_values` | `(6,)` | 六個指定時間的觀察值 |

在常數係數模型下，這種常態增量累積法具有正確的網格點聯合分配；圖上的直線連接僅為視覺呈現，不代表已模擬網格點之間完整的 Brownian 路徑。

## 參數說明

參數直接設定於 `GWP_simulation.py`，修改後重新執行整份程式。

| 參數 | 預設值 | 說明 |
| --- | --- | --- |
| `SEED` | `5566` | NumPy 隨機種子 |
| `T` | `6` | 模擬終點；起點為 `0` |
| `N` | `3000` | 時間區間數，須為正整數 |
| `dt` | `T / N = 0.002` | 時間步長，由前兩項計算 |
| `S_t0` | `2.0` | 初始值 $S_0$ |
| `sigma` | `0.30` | 波動係數，通常取非負值 |
| `observation_times` | `[1, 2, 3, 4, 5, 6]` | 圖上的觀察時間 |

`T` 須大於零。時間單位由使用情境決定；程式沒有指定年、日或秒，解讀 `sigma` 時須使用一致的時間單位。

預設觀察時間對應網格索引 `[500, 1000, 1500, 2000, 2500, 3000]`。圖中的 $t_1,t_2,\ldots,t_n$ 是觀察點的示意標籤；例如圖上的 $t_1$ 代表時間 `1`，不是細網格的第一步 `0.002`。

## 安裝需求

- Python 3。
- NumPy：隨機抽樣、陣列與累積和。
- Matplotlib：路徑繪圖與數學標籤。

以下以 macOS / Linux 的終端機操作為例：

```bash
git clone https://github.com/YangDChen/A-simulation-of-Generalized-Wiener-process.git
cd A-simulation-of-Generalized-Wiener-process
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib
```

Windows PowerShell 的虛擬環境啟用指令為 `.\.venv\Scripts\Activate.ps1`；建立環境時可依安裝方式使用 `python` 或 `py`。

程式未指定最低套件版本。若需保存可重現的環境，請另外記錄 Python 版本與實際安裝版本：

```bash
python --version
python -m pip freeze > requirements-lock.txt
```

## 執行方式

在包含程式的 repository 根目錄執行：

```bash
python GWP_simulation.py
```

也可在 VS Code 或 Jupyter 執行，但請確認所選 interpreter / kernel 已安裝上述套件。程式在頂層直接執行，因此以 `import` 載入也會觸發模擬與繪圖。

## 輸出內容

### 終端機

預設參數下會依序印出：

```text
3000
3001
3001
```

分別是 `dW`、`W`、`S` 的長度。這些是陣列大小檢查，不是統計檢定或模擬準確度指標。

### 圖形

程式透過 `plt.show()` 顯示一張 `12 × 6` 英吋的圖：

- 黑色線條表示 $S_t$ 的單一路徑。
- 起點與六個觀察值以黑點標示。
- 垂直線從 `y=0` 連到觀察值，作為讀圖輔助。
- 第五個觀察點仍會繪製，但其文字與 x 軸刻度標籤留白。

預設不輸出 PNG、PDF、CSV 或觀察值表格；原始程式的 `plt.savefig(...)` 區塊已註解。若需要儲存圖片，可在 `plt.show()` **之前**加入：

```python
plt.savefig("brownian_continuous_discrete.png", dpi=300, bbox_inches="tight")
```

此範例會存至目前工作目錄。若改用子目錄，須先建立該目錄。無圖形介面的環境需選擇適當的非互動繪圖後端並啟用儲存；單靠 `plt.show()` 不會產生圖片檔案。

## Random seed 與可重現性

程式在抽樣前執行 `np.random.seed(5566)`，設定 NumPy 全域隨機狀態。在相同環境、參數、亂數 API 與呼叫順序下，重新執行完整程式可重現同一組衝擊與路徑。

- 更換 `SEED` 會產生另一條路徑，但不改變模型的理論分配。
- Notebook 若只重跑抽樣區塊、未重設種子，會接續既有隨機狀態，得到不同結果。
- 改變抽樣順序、增加先行亂數呼叫或改用 `np.random.default_rng`，即使種子相同也可能產生不同序列。
- 改變 `N` 會改變網格與抽樣；相同種子不代表得到原路徑的單純加密版本。
- 精確重現時應保存程式版本、Python / NumPy / Matplotlib 版本；圖形外觀也可能受字型與繪圖後端影響。

## 注意事項與限制

1. **負值可能被圖形裁切。** 模型允許負值，但目前 y 軸下限固定為 `0`。變更參數後，應檢查 `S` 的最小值並調整顯示範圍；圖上看不到負值不代表模型禁止負值。
2. **觀察值採最近網格點，沒有插值。** 非網格時間會取最近的值，但標記仍畫在原本指定的時間；超出 `[0, T]` 的時間會取到端點，程式未主動報錯。
3. **觀察標籤是固定清單。** 修改觀察點數量時，須同步調整 `S_labels` 與 `ax.set_xticklabels(...)`；縮短 `T` 時也須檢查觀察時間。
4. **單一路徑不等於理論期望。** 零漂移只表示期望值維持 $S_0$，不代表每次模擬都回到起點。程式沒有利用多條路徑估計平均數、變異數或信賴區間。
5. **波動係數只縮放一次。** `W` 已是標準 Brownian motion，再由 `S_t0 + sigma * W` 引入波動係數；不要同時在 `dW` 重複乘上 `sigma`。
