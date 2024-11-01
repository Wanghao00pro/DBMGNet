# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 创建数据
# x = np.linspace(-5, 5, 100)
# y = np.linspace(-5, 5, 100)
# x, y = np.meshgrid(x, y)
# # z = np.sin(np.sqrt(x**2 + y**2))
# z = np.sin(x) * np.cos(y)
# # 创建图形和轴
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')


# # 绘制曲面图
# surface = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# # 隐藏坐标轴
# ax.set_axis_off()
# # 添加颜色条
# # fig.colorbar(surface)

# # 设置标签
# # ax.set_xlabel('Frequency X')
# # ax.set_ylabel('Frequency Y')
# # ax.set_zlabel('Amplitude')

# # 显示图形
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # 模拟参数
# h = 8
# dim = 16

# # 使用numpy初始化权重的实部
# real_weights = np.random.randn(h*h, dim//2 + 1) * 0.02

# # 创建数据网格
# x = np.arange(h*h)
# y = np.arange(dim//2 + 1)
# x, y = np.meshgrid(x, y)

# # 创建图形和轴
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制曲面图
# surface = ax.plot_surface(x, y, real_weights.T, cmap='viridis', edgecolor='none')

# # 隐藏坐标轴
# ax.set_axis_off()

# # 显示图形
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 定义网格和多元正态数据
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
rv = multivariate_normal([0, 0], [[1, 0], [0, 1]])

# 计算Z值
Z = rv.pdf(pos)

# 绘制图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# 隐藏坐标轴
ax.set_axis_off()

# 显示图形
plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import griddata

# Step 1: Generate a hyperspectral image patch
# hs_image = np.random.rand(8, 8, 10)

# # Step 2: Select one spectral band and perform FFT
# spectral_band = hs_image[:, :, 0]  # Select the first spectral band
# fft_result = np.fft.fftshift(np.fft.fft2(spectral_band))

# # Get the magnitude spectrum
# magnitude_spectrum = np.abs(fft_result)

# # Create original grid
# x = np.arange(0, 8)
# y = np.arange(0, 8)
# X, Y = np.meshgrid(x, y)

# # Step 3: Create a higher resolution grid for smoother surface
# x_high_res = np.linspace(0, 7, 100)
# y_high_res = np.linspace(0, 7, 100)
# X_high_res, Y_high_res = np.meshgrid(x_high_res, y_high_res)

# # Interpolate to get a smooth surface
# Z_high_res = griddata((X.flatten(), Y.flatten()), magnitude_spectrum.flatten(), (X_high_res, Y_high_res), method='cubic')

# # Step 4: Plot the smooth surface
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Plot the surface
# ax.plot_surface(X_high_res, Y_high_res, Z_high_res, cmap='viridis', edgecolor='none')

# # Hide axes for a cleaner look
# ax.set_axis_off()

# # Show plot
# plt.show()

