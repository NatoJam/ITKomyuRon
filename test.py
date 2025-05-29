import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# 空間と時間の離散化
x = np.linspace(0, 1, 200)
t_values = [0.0, 0.01, 0.05, 0.1]

# 初期条件 u(0, x) = x(1 - x)
def initial_condition(x):
    return x * (1 - x)

# 拡散方程式の近似解（1〜3次モードのフーリエ展開近似）
def heat_solution(x, t, alpha=1.0):
    u = np.zeros_like(x)
    for n in range(1, 4):
        lam_n = (n * np.pi) ** 2
        phi_n = np.sin(n * np.pi * x)
        a_n = 2 * np.trapezoid(initial_condition(x) * phi_n, x)
        u += a_n * np.exp(-alpha * lam_n * t) * phi_n
    return u

# グラフ作成
fig, ax = plt.subplots(figsize=(10, 6))
canvas = FigureCanvas(fig)  # ← ここで FigureCanvasAgg を使用！

for t in t_values:
    u_xt = heat_solution(x, t)
    ax.plot(x, u_xt, label=f't = {t:.2f}')

ax.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax.text(1.01, 0.1, 'x = 1\n(観測点)', verticalalignment='center')
ax.set_title('時間発展における温度分布と右端の勾配 $u_x(t,1)$ の挙動')
ax.set_xlabel('空間位置 $x$')
ax.set_ylabel('温度 $u(t,x)$')
ax.legend()
ax.grid(True)
fig.tight_layout()

# 描画とRGB画像の取得
canvas.draw()
width, height = canvas.get_width_height()

# ARGBバッファ取得 → reshape（4チャンネル）
image = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
image = image.reshape((height, width, 4))

# RGBに変換したい場合
image_rgb = image[:, :, 1:]  # R,G,Bチャンネルのみ取り出す
image_rgb = np.concatenate((image_rgb[:, :, 2:], image_rgb[:, :, :1]), axis=2)  # BGRからRGBに変換

from PIL import Image

# RGB画像をPILに渡して表示
img_pil = Image.fromarray(image_rgb)
img_pil.show()

# image は (H, W, 3) のNumPy配列として取得済み → OpenCV, PIL等に渡せます