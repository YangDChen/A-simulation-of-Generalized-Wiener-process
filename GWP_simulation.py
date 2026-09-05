import numpy as np
import matplotlib.pyplot as plt

##### 參數設定 #####
SEED = 5566
np.random.seed(SEED)

T = 6 #總時間區間 \in [0, 6]
N = 3000 #再切3,000個子時間區間
dt = T / N #每一個子時間區間長度 == 6/3,000 == 2e-3
t = np.linspace(0, T, N + 1) #3,001個時間點

#一般化的Wiener process Model: S_t = S_{t_0} + \sigma * W_t
S_t0 = 2.0 #隨機過程的起始值
sigma = 0.30 #隨機過程的波動度

##### 模擬Brownian motion: Part I #####

# e_j i.i.d. N(0, 1) \forall j \in [0, T]
e = np.random.normal(
	loc=0,
	scale=1,
	size=N
)

# Standard Brownian increment滿足：
# 1. W_t - W_s ~ N(0, t-s), 0 <= s < t
# 2. 對於一個長度為\Delta t(code以dt代稱)的子時間區間\Delta W_i而言：
	# a. \Delta W_i = W_{t_i} - W_{t_{i-1}}, \forall i \in [0, T] 
	# b. \Delta W_i ~ N(0, \Delta t) (其中\Delta W_i在code中以dW代稱)
		# i. 動差性質請參見性質1.1的條件2

#利用shock: e_j i.i.d. N(0, 1)建立子時間區間 \Delta W_i (dW)
# dW = \sqrt(dt) * e ~ N(0, dt)
	# <=> \Delta W_i = \sqrt(\Delta t) * e_j ~ N(0, \Delta t)
	#其中 e i.i.d. N(0, 1)

### proof ###
# 1. E(dW) = \sqrt(dt)E(e) = 0
# 2. Var(dW) = dt * Var(e) = dt == 2e-3

#結論：dW ~ N(0, dt), 其中dt是每個子時間區間的長度 == 2e-3
#這個是子時間區間
dW = np.sqrt(dt) * e

print(len(dW)) #要 == N == 3,000

##### 模擬Brownian motion: Part II #####
#這個是時間點
#把起始值和\Delta W弄成一串array, 其中起始值是scalar, \Delta W是vector
# W_0 = 0 (Brownian motion的起始值)
# np.cunsum()是累積和 e.g., a =[1, 2, 3], np.cumsum(a) = [1, 3, 6]
W = np.concatenate([
	[0],
	np.cumsum(dW)
])

print(len(W)) #要 == N+1 == 3,001

#建構一般化的Wiener process Model: 
#Definition: DX_t = \mu * dt + \sigma * dW_t, where W_t為Standard Wiener process
# Model:S_t = S_{t_0} + \sigma * W_t
# S_t0 == 2 (scalar) ; sigma == 0.3 (scalar) ; W (array)
S = S_t0 + sigma * W

print(len(S)) #也要 == 3,001

##### X軸的tick和對應value的tick #####
observation_times = np.array([
	1, 2, 3, 4, 5, 6
])

#t - time是每個模擬時間點與目標時間的距離,
	#例如t=[0, 0.002, 0.004, ..., 5.998, 6.0], time=1, 則t - time = [-1, -0.998, -0.996, ..., 4.998, 5.0]
#np.argmin(np.abs(t - time))會回傳距離最小的模擬時間點的index
observation_indices = []
  
for time in observation_times:
	index = np.argmin(np.abs(t - time))
	observation_indices.append(index)

observation_values = S[observation_indices]

##### 畫圖 #####
#fig = figure ; ax = axes
fig, ax = plt.subplots(figsize=(12, 6))
  
#畫一般化的Wiener process value
#(x, y) = (t, S)
#zorder=1表示這條線在圖層的最底層, 數字越大會畫在越上層
ax.plot(
	t,
	S,
	color="black",
	linewidth=0.9,
	zorder=1
)

#畫X軸t_i對應到value的S_{t_i}的垂直線, \forall i = {1, 2, 3, 4, n}
#zip()把兩個序列配對，例如a = [1, 2, 3], b = [4, 5, 6]，zip(a, b)會得到[(1, 4), (2, 5), (3, 6)]

#每次迴圈會讓time = observation_times[i], value = observation_values[i] (觀察時間對應的隨機路徑值)
for time, value in zip(observation_times, observation_values):
	ax.vlines(
	x=time, #垂直線的x座標是觀察時間
	ymin=0, #垂直線的y座標從0開始
	ymax=value, #垂直線從0開始畫到該觀察時間對應的隨機路徑值 (所以每個觀察時間會有一條垂直線 0 -> S_{t_i})
	color="black",
	linewidth=0.6,
	zorder=1
)

#畫離散觀察點
ax.scatter(
	observation_times,
	observation_values,
	color="black",
	s=32, #marker size
	zorder=4
)

#建立label: S_{t_i}的list，對應每個觀察時間的隨機路徑值
#r"..."表示raw string，避免轉義字符的影響"
S_labels = [
	r"$S_{t_1}$",
	r"$S_{t_2}$",
	r"$S_{t_3}$",
	r"$S_{t_4}$",
	"", #第五個觀察時間不需要標籤，所以用空字串表示
	r"$S_{t_n}$"
]

#把label: S_{t_i}放到time對應的value上
for time, value, label in zip(observation_times, observation_values, S_labels):
	if label != "":
		ax.annotate(
		label,
		xy=(time, value), #指定要標注的資料點座標
		xytext=(0, 15), #標注文字相較於資料點的位置 (水平移動=0, 垂直移動=向上移15個點)
		textcoords="offset points", #告訴Matplotlib標注文字的座標是相對於資料點的偏移量，不是新的資料座標
		ha="center", #水平置中 ha=horizontal alignment
		va="bottom", #垂直置底 va=vertical alignment
		fontsize=13,
		color="black",
		zorder=5,
		bbox=dict( #bbox是文字的背景框
		facecolor="white", #背景=白色
		edgecolor="none", #邊框=無
		pad=1.5 #文字與背景框之間的空間=1.5個點
		)
	)

#指定x軸的tick位置為觀察時間, 其中觀察時間=observation_times=[1, 2, 3, 4, 5, 6]
ax.set_xticks(observation_times)

#把實際數字[1, 2, 3, 4, 5, 6]換成對應的LaTeX標籤, 第五個空白
ax.set_xticklabels([
	r"$t_1$",
	r"$t_2$",
	r"$t_3$",
	r"$t_4$",
	"",
	r"$t_n$"
	], fontsize=13
)

#指定y的tick位置為S_{t_0}=2.0，也就是隨機過程的起始值 (只設定一個)
ax.set_yticks([S_t0])

#把實際數字2.0換成對應的LaTeX標籤S_{t_0}
ax.set_yticklabels([r"$S_{t_0}$"], fontsize=13)

#隨機過程的起始值畫一個黑點, 座標=(0, S_{t_0}), 其中前面已定義S_{t_0}=2.0
ax.scatter(
	[0],
	[S_t0],
	color="black",
	s=32,
	zorder=4
)

##### X, Y軸的label #####

#X軸名稱=t
ax.set_xlabel(r"$t$", fontsize=14)

#Y軸名稱=S_t, 且旋轉角度=0 (水平)
ax.set_ylabel(r"$S_t$", fontsize=14, rotation=0)

##### X, Y軸的appearance #####
#設定X軸範圍 \in [0, T+0.3]，T=6，所以範圍是[0, 6.3] 多留0.3是為了讓圖形看起來不會太擠
ax.set_xlim(0, T + 0.3)

#設定Y軸範圍 \in [0, max(S)+0.5]，max(S)是隨機路徑的最大值，多留0.5是為了讓圖形看起來不會太擠
ax.set_ylim(0, max(S) + 0.5)

#spines是Matplotlib中用來表示圖形邊框的物件，通常有四個spines: top, bottom, left, right

#隱藏上邊框
ax.spines["top"].set_visible(False)

#隱藏右邊框
ax.spines["right"].set_visible(False)

#設定下邊框的位置為y=0，這樣x軸就會在y=0的位置 data表示使用資料座標系統，0表示y=0的位置
ax.spines["bottom"].set_position(
	("data", 0)
)

#設定左邊框的位置為x=0，這樣y軸就會在x=0的位置 data表示使用資料座標系統，0表示x=0的位置
ax.spines["left"].set_position(
	("data", 0)
)

ax.spines["bottom"].set_color("black")
ax.spines["left"].set_color("black")
ax.spines["bottom"].set_linewidth(1.2)
ax.spines["left"].set_linewidth(1.2)
ax.tick_params(
	axis="both", #同時設定x軸和y軸的刻度
	colors="black"
)
plt.tight_layout() #自動調整排版(座標軸、標籤、圖邊界)，避免文字被圖片邊界裁掉
# plt.savefig(
	# "./Proj/Financial_Engineering/brownian_continuous_discrete.png",
	# dpi=300, #圖片解析度=300dpi
	# bbox_inches="tight" #儲存圖片時，自動調整邊界，避免圖片被裁掉
# )
plt.show()