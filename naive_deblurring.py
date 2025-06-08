import numpy as np
import cv2
from skimage import data, color
from skimage.metrics import peak_signal_noise_ratio as PSNR, structural_similarity as SSIM
import matplotlib.pyplot as plt

# 裁剪图像
def crop_center(img, cropx, cropy):
    h, w = img.shape[:2]
    startx = w // 2 - (cropx // 2)
    starty = h // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


camera = data.camera()
coins = data.coins()
astronaut = data.astronaut()

# 转为 256×256 灰度图（这里提供了camera、coins、astro的图片的不同转换，若需要直接改变下面的orig即可）
camera_crop = crop_center(camera, 256, 256).astype(np.float32) / 255.0
coins_crop = crop_center(coins, 256, 256).astype(np.float32) / 255.0
astronaut_gray = color.rgb2gray(astronaut)
astro_crop = crop_center((astronaut_gray * 255).astype(np.uint8), 256, 256).astype(np.float32) / 255.0

# 选择 camera_crop 作为示例，也可以改成 coins_crop 或 astro_crop
orig = camera_crop

# PSF生成
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
             color=1, thickness=1)
    psf /= psf.sum()
    return psf

# 裁剪
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
    W = np.conj(H) / (np.abs(H)**2 + K)
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

    W = np.conj(H) / (np.abs(H)**2 + alpha * np.abs(L)**2)
    F_hat = W * G
    f_est = np.real(np.fft.ifft2(F_hat))
    return f_est[:orig_shape[0], :orig_shape[1]]

# 计算 PSNR 和 SSIM 
def compute_metrics(orig, restored):
    psnr_val = PSNR(orig, restored, data_range=orig.max() - orig.min())
    ssim_val = SSIM(orig, restored, data_range=orig.max() - orig.min())
    return psnr_val, ssim_val

# Sobel/Laplacian 算子度量边缘强度能量 
def sobel_energy(img):
    """
    计算一张归一化灰度图（0-1）在 Sobel 梯度下的平均梯度幅值。
    """
    img8 = (img * 255).astype(np.uint8)
    gx = cv2.Sobel(img8, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img8, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return np.mean(mag)

def laplacian_energy(img):
    """
    计算一张归一化灰度图（0-1）在 Laplacian 算子的平均绝对值。
    """
    img8 = (img * 255).astype(np.uint8)
    lap = cv2.Laplacian(img8, cv2.CV_64F)
    return np.mean(np.abs(lap))

# 生成运动模糊+噪声
 
orig = camera_crop
 
L = 10        # 运动长度
theta = 30    # 方向
psf_motion = motion_psf(L, theta, size=2*L+1)

# 生成纯“运动模糊”图
blurred = convolve_fft(orig, psf_motion)

# 加噪声
noise_sigma = 0.02
noisy_blurred = blurred + np.random.randn(*blurred.shape) * noise_sigma
noisy_blurred = np.clip(noisy_blurred, 0, 1)

# 计算 PSNR 和 SSIM
blur_psnr, blur_ssim = compute_metrics(orig, blurred)
noisy_psnr, noisy_ssim = compute_metrics(orig, noisy_blurred)

e_blur_sobel      = sobel_energy(blurred)
e_blur_lap        = laplacian_energy(blurred)
e_noisy_sobel     = sobel_energy(noisy_blurred)
e_noisy_lap       = laplacian_energy(noisy_blurred)

inv_res = inverse_filtering(noisy_blurred, psf_motion, eps=1e-3)
wnr_res = wiener_filtering(noisy_blurred, psf_motion, K=1e-2)
tik_res = tikhonov_filtering(noisy_blurred, psf_motion, alpha=1e-2)

# 计算恢复后 PSNR, SSIM
inv_psnr, inv_ssim = compute_metrics(orig, inv_res)
wnr_psnr, wnr_ssim = compute_metrics(orig, wnr_res)
tik_psnr, tik_ssim = compute_metrics(orig, tik_res)

'''
# 计算恢复图 Sobel/Laplacian 能量
e_inv_sobel  = sobel_energy(inv_res)
e_inv_lap    = laplacian_energy(inv_res)
e_wnr_sobel  = sobel_energy(wnr_res)
e_wnr_lap    = laplacian_energy(wnr_res)
e_tik_sobel  = sobel_energy(tik_res)
e_tik_lap    = laplacian_energy(tik_res)
'''

print(f"运动模糊 -> PSNR: {blur_psnr:.2f} dB, SSIM: {blur_ssim:.4f}, "
      f"Sobel 能量: {e_blur_sobel:.2f}, Laplacian 能量: {e_blur_lap:.2f}")
print(f"运动模糊 + 噪声 -> PSNR: {noisy_psnr:.2f} dB, SSIM: {noisy_ssim:.4f}, "
      f"Sobel 能量: {e_noisy_sobel:.2f}, Laplacian 能量: {e_noisy_lap:.2f}")
print(f"Inverse -> PSNR: {inv_psnr:.2f} dB, SSIM: {inv_ssim:.4f}, "
      f"Sobel 能量: {e_inv_sobel:.2f}, Laplacian 能量: {e_inv_lap:.2f}")
print(f"Wiener  -> PSNR: {wnr_psnr:.2f} dB, SSIM: {wnr_ssim:.4f}, "
      f"Sobel 能量: {e_wnr_sobel:.2f}, Laplacian 能量: {e_wnr_lap:.2f}")
print(f"Tikhonov-> PSNR: {tik_psnr:.2f} dB, SSIM: {tik_ssim:.4f}, "
      f"Sobel 能量: {e_tik_sobel:.2f}, Laplacian 能量: {e_tik_lap:.2f}")


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 3, figsize=(12, 8))


axes[0, 0].imshow(orig, cmap='gray')
axes[0, 0].set_title('原始清晰图')
axes[0, 0].axis('off')

axes[0, 1].imshow(blurred, cmap='gray')
axes[0, 1].set_title(f'运动模糊\nPSNR={blur_psnr:.2f} SSIM={blur_ssim:.3f}')
axes[0, 1].axis('off')

axes[0, 2].imshow(noisy_blurred, cmap='gray')
axes[0, 2].set_title(f'运动模糊+噪声\nPSNR={noisy_psnr:.2f} SSIM={noisy_ssim:.3f}')
axes[0, 2].axis('off')

axes[1, 0].imshow(inv_res, cmap='gray')
axes[1, 0].set_title(f'Inverse\nPSNR={inv_psnr:.2f} SSIM={inv_ssim:.3f}')
axes[1, 0].axis('off')

axes[1, 1].imshow(wnr_res, cmap='gray')
axes[1, 1].set_title(f'Wiener\nPSNR={wnr_psnr:.2f} SSIM={wnr_ssim:.3f}')
axes[1, 1].axis('off')

axes[1, 2].imshow(tik_res, cmap='gray')
axes[1, 2].set_title(f'Tikhonov\nPSNR={tik_psnr:.2f} SSIM={tik_ssim:.3f}')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

