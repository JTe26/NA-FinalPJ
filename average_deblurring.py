import numpy as np
import cv2
from skimage import data, color
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
import matplotlib.pyplot as plt


# 裁剪
def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]



camera = data.camera()  # 512×512 灰度
coins = data.coins()  # 303×384 灰度
astronaut = data.astronaut()  # 512×512 彩色

# 转为 256×256 灰度图，并归一化到 [0,1]
camera_crop = crop_center(camera, 256, 256).astype(np.float32) / 255.0
coins_crop = crop_center(coins, 256, 256).astype(np.float32) / 255.0
astronaut_gray = color.rgb2gray(astronaut)
astro_crop = crop_center((astronaut_gray * 255).astype(np.uint8), 256, 256).astype(np.float32) / 255.0

# 选择 camera_crop 作为示例，也可以改成 coins_crop 或 astro_crop
orig = camera_crop


# 生成运动模糊 PSF（理想线段核）
def motion_psf(L, theta_deg, size):
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    theta = np.deg2rad(theta_deg)
    x0 = center - (L - 1) / 2 * np.cos(theta)
    y0 = center - (L - 1) / 2 * np.sin(theta)
    x1 = center + (L - 1) / 2 * np.cos(theta)
    y1 = center + (L - 1) / 2 * np.sin(theta)
    cv2.line(psf,
             (int(round(x0)), int(round(y0))),
             (int(round(x1)), int(round(y1))),
             color=1,
             thickness=1)
    psf /= psf.sum()
    return psf


# 多帧平均模拟运动模糊
def simulate_motion_blur_avg(im, L, theta_deg, num_frames):
    """
    使用均匀多帧平均的方式模拟运动模糊：
    - im: 输入灰度图像（浮点归一化）
    - L: 总移动像素距离
    - theta_deg: 运动方向，度为单位
    - num_frames: 在运动过程中均匀采样帧数
    """
    h, w = im.shape
    blurred = np.zeros_like(im)
    theta = np.deg2rad(theta_deg)

    for i in range(num_frames):
        t = i / (num_frames - 1)  # 从 0 到 1 均匀
        dx = t * L * np.cos(theta)
        dy = t * L * np.sin(theta)
        M = np.float32([[1, 0, dx],
                        [0, 1, dy]])
        shifted = cv2.warpAffine(im, M, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        blurred += shifted

    blurred /= num_frames
    return blurred


# 多帧平均反推实际 PSF
def build_actual_psf(L, theta_deg, num_frames, size):

    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    theta = np.deg2rad(theta_deg)

    for i in range(num_frames):
        t = i / (num_frames - 1)
        x_shift = center + t * L * np.cos(theta)
        y_shift = center + t * L * np.sin(theta)

        # 四个最近整数坐标
        x0, y0 = int(np.floor(x_shift)), int(np.floor(y_shift))
        wx, wy = x_shift - x0, y_shift - y0

        # 累加双线性插值权重
        if 0 <= x0 < size and 0 <= y0 < size:
            psf[y0, x0] += (1 - wx) * (1 - wy)
        if 0 <= x0 + 1 < size and 0 <= y0 < size:
            psf[y0, x0 + 1] += wx * (1 - wy)
        if 0 <= x0 < size and 0 <= y0 + 1 < size:
            psf[y0 + 1, x0] += (1 - wx) * wy
        if 0 <= x0 + 1 < size and 0 <= y0 + 1 < size:
            psf[y0 + 1, x0 + 1] += wx * wy

    psf /= num_frames
    return psf


=
def convolve_fft(im, kernel):
    M, N = im.shape
    P, Q = kernel.shape
    pad_shape = (M + P - 1, N + Q - 1)

    im_pad = np.zeros(pad_shape, dtype=np.float32)
    im_pad[:M, :N] = im

    ker_shift = np.fft.ifftshift(kernel)
    ker_pad = np.zeros(pad_shape, dtype=np.float32)
    ker_pad[:P, :Q] = ker_shift

    G = np.fft.fft2(im_pad)
    H = np.fft.fft2(ker_pad)
    conv = np.real(np.fft.ifft2(G * H))

    return conv[:M, :N]


# 已知psf去卷积
def _fft_with_psf(im, psf):
    M, N = im.shape
    P, Q = psf.shape
    pad_shape = (M + P - 1, N + Q - 1)

    im_pad = np.zeros(pad_shape, dtype=np.float32)
    im_pad[:M, :N] = im
    G = np.fft.fft2(im_pad)

    psf_shift = np.fft.ifftshift(psf)
    ker_pad = np.zeros(pad_shape, dtype=np.float32)
    ker_pad[:P, :Q] = psf_shift
    H = np.fft.fft2(ker_pad)

    return G, H, pad_shape, (M, N)


# 直接反卷积
def inverse_filtering(blurred, psf, eps=1e-3):
    G, H, pad_shape, orig_shape = _fft_with_psf(blurred, psf)
    W = np.where(np.abs(H) > eps, 1.0 / H, 0.0)
    F_hat = W * G
    f_est = np.real(np.fft.ifft2(F_hat))
    return f_est[:orig_shape[0], :orig_shape[1]]


# Wiener
def wiener_filtering(blurred, psf, K=1e-2):
    G, H, pad_shape, orig_shape = _fft_with_psf(blurred, psf)
    W = np.conj(H) / (np.abs(H) ** 2 + K)
    F_hat = W * G
    f_est = np.real(np.fft.ifft2(F_hat))
    return f_est[:orig_shape[0], :orig_shape[1]]


# Tikhonov
def tikhonov_filtering(blurred, psf, alpha=1e-3):
    G, H, pad_shape, orig_shape = _fft_with_psf(blurred, psf)
    lap = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]], dtype=np.float32)
    P, Q = lap.shape
    M_pad, N_pad = pad_shape
    L_pad = np.zeros((M_pad, N_pad), dtype=np.float32)
    lap_shift = np.fft.ifftshift(lap)
    L_pad[:P, :Q] = lap_shift
    L = np.fft.fft2(L_pad)

    W = np.conj(H) / (np.abs(H) ** 2 + alpha * np.abs(L) ** 2)
    F_hat = W * G
    f_est = np.real(np.fft.ifft2(F_hat))
    return f_est[:orig_shape[0], :orig_shape[1]]


# ========== 11. 计算 PSNR 和 SSIM ==========
def compute_metrics(orig, restored):
    psnr_val = PSNR(orig, restored, data_range=orig.max() - orig.min())
    ssim_val = SSIM(orig, restored, data_range=orig.max() - orig.min())
    return psnr_val, ssim_val


# ========== 12. Sobel/Laplacian 算子度量边缘强度能量 ==========
def sobel_energy(img):
    """
    计算一张归一化灰度图（0-1）在 Sobel 梯度下的平均梯度幅值。
    """
    img8 = (img * 255).astype(np.uint8)
    gx = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return np.mean(mag)


def laplacian_energy(img):
    """
    计算一张归一化灰度图（0-1）在 Laplacian 算子的平均绝对值。
    """
    img8 = (img * 255).astype(np.uint8)
    lap = cv2.Laplacian(img8, cv2.CV_64F)
    return np.mean(np.abs(lap))


# =============================================================================
# ===               主程序示例：生成运动模糊 + 去卷积          ===
# =============================================================================

# 选择一张裁剪后的测试图
orig = camera_crop

# 设置运动模糊参数
L = 10  # 总共平移 10 像素
theta = 30  # 运动方向 30 度
num_frames = 40  # 多帧平均的帧数

# 生成“多帧平均”方式的运动模糊图
blurred = simulate_motion_blur_avg(orig, L, theta, num_frames)

# 在运动模糊后再加噪声
noise_sigma = 0.02
noisy_blurred = blurred + np.random.randn(*blurred.shape) * noise_sigma
noisy_blurred = np.clip(noisy_blurred, 0, 1)

# 计算 PSNR 和 SSIM（无噪声 & 有噪声）
blur_psnr, blur_ssim = compute_metrics(orig, blurred)
noisy_psnr, noisy_ssim = compute_metrics(orig, noisy_blurred)

# 计算 Sobel/Laplacian 能量（无噪声 & 有噪声）
e_blur_sobel = sobel_energy(blurred)
e_blur_lap = laplacian_energy(blurred)
e_noisy_sobel = sobel_energy(noisy_blurred)
e_noisy_lap = laplacian_energy(noisy_blurred)

# ========== 13. 构造“真实”PSF（插值叠加）用于去卷积 ==========
psf_size = 2 * L + 1
actual_psf = build_actual_psf(L, theta, num_frames, size=psf_size)

# ========== 14. 对比：如果仍用理想线段核 vs 用实际插值核做去卷积 ==========
# 理想线段核
ideal_psf = motion_psf(L, theta, size=psf_size)

# 先用理想 PSF 去卷积
inv_res_ideal = inverse_filtering(noisy_blurred, ideal_psf, eps=1e-3)
wnr_res_ideal = wiener_filtering(noisy_blurred, ideal_psf, K=1e-2)
tik_res_ideal = tikhonov_filtering(noisy_blurred, ideal_psf, alpha=1e-2)

# 再用“实际”PSF 去卷积
inv_res_real = inverse_filtering(noisy_blurred, actual_psf, eps=1e-3)
wnr_res_real = wiener_filtering(noisy_blurred, actual_psf, K=1e-2)
tik_res_real = tikhonov_filtering(noisy_blurred, actual_psf, alpha=1e-2)

# 计算恢复后 PSNR & SSIM
inv_psnr_ideal, inv_ssim_ideal = compute_metrics(orig, inv_res_ideal)
wnr_psnr_ideal, wnr_ssim_ideal = compute_metrics(orig, wnr_res_ideal)
tik_psnr_ideal, tik_ssim_ideal = compute_metrics(orig, tik_res_ideal)

inv_psnr_real, inv_ssim_real = compute_metrics(orig, inv_res_real)
wnr_psnr_real, wnr_ssim_real = compute_metrics(orig, wnr_res_real)
tik_psnr_real, tik_ssim_real = compute_metrics(orig, tik_res_real)

e_inv_sobel_ideal = sobel_energy(inv_res_ideal)
e_inv_lap_ideal = laplacian_energy(inv_res_ideal)
e_wnr_sobel_ideal = sobel_energy(wnr_res_ideal)
e_wnr_lap_ideal = laplacian_energy(wnr_res_ideal)
e_tik_sobel_ideal = sobel_energy(tik_res_ideal)
e_tik_lap_ideal = laplacian_energy(tik_res_ideal)

e_inv_sobel_real = sobel_energy(inv_res_real)
e_inv_lap_real = laplacian_energy(inv_res_real)
e_wnr_sobel_real = sobel_energy(wnr_res_real)
e_wnr_lap_real = laplacian_energy(wnr_res_real)
e_tik_sobel_real = sobel_energy(tik_res_real)
e_tik_lap_real = laplacian_energy(tik_res_real)

print("===== 理想线段 PSF 去卷积 =====")
print(f"Inverse(ideal) -> PSNR: {inv_psnr_ideal:.2f} dB, SSIM: {inv_ssim_ideal:.4f}, "
      f"Sobel 能量: {e_inv_sobel_ideal:.2f}, Laplacian 能量: {e_inv_lap_ideal:.2f}")
print(f"Wiener(ideal)  -> PSNR: {wnr_psnr_ideal:.2f} dB, SSIM: {wnr_ssim_ideal:.4f}, "
      f"Sobel 能量: {e_wnr_sobel_ideal:.2f}, Laplacian 能量: {e_wnr_lap_ideal:.2f}")
print(f"Tikhonov(ideal)-> PSNR: {tik_psnr_ideal:.2f} dB, SSIM: {tik_ssim_ideal:.4f}, "
      f"Sobel 能量: {e_tik_sobel_ideal:.2f}, Laplacian 能量: {e_tik_lap_ideal:.2f}")

print("\n===== 实际插值 PSF 去卷积 =====")
print(f"Inverse(real) -> PSNR: {inv_psnr_real:.2f} dB, SSIM: {inv_ssim_real:.4f}, "
      f"Sobel 能量: {e_inv_sobel_real:.2f}, Laplacian 能量: {e_inv_lap_real:.2f}")
print(f"Wiener(real)  -> PSNR: {wnr_psnr_real:.2f} dB, SSIM: {wnr_ssim_real:.4f}, "
      f"Sobel 能量: {e_wnr_sobel_real:.2f}, Laplacian 能量: {e_wnr_lap_real:.2f}")
print(f"Tikhonov(real)-> PSNR: {tik_psnr_real:.2f} dB, SSIM: {tik_ssim_real:.4f}, "
      f"Sobel 能量: {e_tik_sobel_real:.2f}, Laplacian 能量: {e_tik_lap_real:.2f}")

print("\n===== 运动模糊前 & 多帧平均 & 加噪声 =====")
print(f"运动模糊 -> PSNR: {blur_psnr:.2f} dB, SSIM: {blur_ssim:.4f}, "
      f"Sobel 能量: {e_blur_sobel:.2f}, Laplacian 能量: {e_blur_lap:.2f}")
print(f"模糊+噪声 -> PSNR: {noisy_psnr:.2f} dB, SSIM: {noisy_ssim:.4f}, "
      f"Sobel 能量: {e_noisy_sobel:.2f}, Laplacian 能量: {e_noisy_lap:.2f}")

# Visualization
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(3, 3, figsize=(16, 12))

axes[0, 0].imshow(orig, cmap='gray')
axes[0, 0].set_title('原始清晰图')
axes[0, 0].axis('off')

axes[0, 1].imshow(blurred, cmap='gray')
axes[0, 1].set_title(f'多帧平均运动模糊')
axes[0, 1].axis('off')

axes[0, 2].imshow(noisy_blurred, cmap='gray')
axes[0, 2].set_title(f'多帧平均运动模糊加噪声\nPSNR={blur_psnr:.2f} SSIM={blur_ssim:.3f}')
axes[0, 2].axis('off')

axes[1, 0].imshow(inv_res_ideal, cmap='gray')
axes[1, 0].set_title(f'Inverse(ideal)\nPSNR={inv_psnr_ideal:.2f} SSIM={inv_ssim_ideal:.3f}')
axes[1, 0].axis('off')

axes[1, 1].imshow(wnr_res_ideal, cmap='gray')
axes[1, 1].set_title(f'Wiener(ideal)\nPSNR={wnr_psnr_ideal:.2f} SSIM={wnr_ssim_ideal:.3f}')
axes[1, 1].axis('off')

axes[1, 2].imshow(tik_res_ideal, cmap='gray')
axes[1, 2].set_title(f'Tikhonov(ideal)\nPSNR={tik_psnr_ideal:.2f} SSIM={tik_ssim_ideal:.3f}')
axes[1, 2].axis('off')

axes[2, 0].imshow(inv_res_real, cmap='gray')
axes[2, 0].set_title(f'Inverse(real)\nPSNR={inv_psnr_real:.2f} SSIM={inv_ssim_real:.3f}')
axes[2, 0].axis('off')

axes[2, 1].imshow(wnr_res_real, cmap='gray')
axes[2, 1].set_title(f'Wiener(real)\nPSNR={wnr_psnr_real:.2f} SSIM={wnr_ssim_real:.3f}')
axes[2, 1].axis('off')

axes[2, 2].imshow(tik_res_real, cmap='gray')
axes[2, 2].set_title(f'Tikhonov(real)\nPSNR={tik_psnr_real:.2f} SSIM={tik_ssim_real:.3f}')
axes[2, 2].axis('off')



plt.tight_layout()
plt.show()
