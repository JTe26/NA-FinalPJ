import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import cv2
from skimage.transform import radon

orig = data.camera().astype(np.float32) / 255.0


# 均匀多帧平均模拟运动模糊
def simulate_motion_blur_avg(im, L, theta_deg, num_frames):

    h, w = im.shape
    blurred = np.zeros_like(im)
    theta = np.deg2rad(theta_deg)

    for i in range(num_frames):
        t = i / (num_frames - 1)  # 从 0 到 1 均匀
        dx = t * L * np.cos(theta)
        dy = t * L * np.sin(theta)
        # 生成仿射平移矩阵
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        # 使用双线性插值，边界填充 0
        shifted = cv2.warpAffine(im, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        blurred += shifted

    blurred /= num_frames
    return blurred


L = 10
theta = 30 
num_frames = 40 

# 生成运动模糊图像
blurred_avg = simulate_motion_blur_avg(orig, L, theta, num_frames)

# 频谱
F = np.fft.fft2(blurred_avg)
F_shift = np.fft.fftshift(F)
magnitude_spectrum = np.log1p(np.abs(F_shift))


# visualization
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(blurred_avg, cmap='gray')
plt.title('多帧平均模拟的运动模糊图像')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('运动模糊图像频谱 (Log 幅度)')
plt.axis('off')

plt.tight_layout()
plt.show()

# 计算theta
H, W = magnitude_spectrum.shape
cx, cy = W//2, H//2
r = min(cx, cy)
Y, X = np.ogrid[:H, :W]
mask = (X-cx)**2 + (Y-cy)**2 <= r**2
mag_masked = np.zeros_like(magnitude_spectrum)
mag_masked[mask] = magnitude_spectrum[mask]
mag_shifted = mag_masked.copy()

theta_range = np.linspace(0, 180, 181, endpoint=True)
sinogram = radon(mag_masked, theta=theta_range, circle=True)
proj_var = np.var(sinogram, axis=0)
phi = theta_range[np.argmax(proj_var)]
theta_hat = (180-phi) % 180


print( theta_hat)
