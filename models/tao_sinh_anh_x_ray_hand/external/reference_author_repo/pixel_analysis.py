import glob, cv2, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

orig_dir  = r"C:\Projetos\hand-bone\boneage-test-dataset\boneage-test-dataset"
clean_dir = r"C:\Projetos\hand-bone\output_cleaned_mult"
save_path = r"C:\Projetos\hand-bone\figs\pixel_intensity_distribution_mult.png"

# Use int64 for compatibility with np.bincount
hist_orig  = np.zeros(256, dtype=np.int64)
hist_clean = np.zeros(256, dtype=np.int64)

std_orig_list = []
std_clean_list = []

# Loop through original images
for fp_orig in glob.glob(f"{orig_dir}/*.png"):
    case_id = os.path.splitext(os.path.basename(fp_orig))[0]
    img_o = cv2.imread(fp_orig, cv2.IMREAD_GRAYSCALE)
    if img_o is None:
        continue

    # Process the three cleaned images
    for i in range(1, 4):
        fp_clean = os.path.join(clean_dir, case_id, f"{case_id}_cleaned_{i}.png")
        img_c = cv2.imread(fp_clean, cv2.IMREAD_GRAYSCALE)
        if img_c is None:
            continue
        
        # Accumulate histograms
        hist_orig  += np.bincount(img_o.ravel(), minlength=256)
        hist_clean += np.bincount(img_c.ravel(), minlength=256)
        
        # Compute and store std devs
        std_orig_list.append(np.std(img_o))
        std_clean_list.append(np.std(img_c))

# ---- Statistics ----
mean_orig = np.mean(std_orig_list)
std_orig = np.std(std_orig_list)
mean_clean = np.mean(std_clean_list)
std_clean = np.std(std_clean_list)

print(f"Original images: mean std = {mean_orig:.2f} ± {std_orig:.2f}")
print(f"Inpainted images:  mean std = {mean_clean:.2f} ± {std_clean:.2f}")

# ---- Densities ----
dens_orig  = hist_orig  / hist_orig.sum()
dens_clean = hist_clean / hist_clean.sum()
dens_orig_sm  = gaussian_filter1d(dens_orig , sigma=1.5)
dens_clean_sm = gaussian_filter1d(dens_clean, sigma=1.5)

# ---- Plotting ----
sns.set_theme(style="whitegrid", font_scale=1.2)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2.5, 1]})

# Panel 1: Pixel intensity histograms
bins = np.arange(256)
axes[0].bar(bins, dens_orig , width=1, color="steelblue", alpha=0.4, label="Original")
axes[0].bar(bins, dens_clean, width=1, color="tomato"   , alpha=0.4, label="Inpainted")
axes[0].plot(bins, dens_orig_sm , color="steelblue", linewidth=1.8)
axes[0].plot(bins, dens_clean_sm, color="tomato"   , linewidth=1.8)
axes[0].set_xlim(0, 255)
axes[0].set_xlabel("Pixel Intensity (0–255)")
axes[0].set_ylabel("Density")
axes[0].set_title("Pixel Intensity Distribution")
axes[0].legend()

# Panel 2: Boxplot of std deviations
sns.boxplot(data=[std_orig_list, std_clean_list],
            palette=["steelblue", "tomato"],
            width=0.5,
            ax=axes[1])
axes[1].set_xticklabels(["Original", "Inpainted"])
axes[1].set_ylabel("Pixel Intensity Std. Dev.")
axes[1].set_title("Image-wise Noise")

# Text annotation
axes[1].text(0, mean_orig + 0.5, f"{mean_orig:.2f} ± {std_orig:.2f}", 
             ha='center', fontsize=11, color="black")
axes[1].text(1, mean_clean + 0.5, f"{mean_clean:.2f} ± {std_clean:.2f}", 
             ha='center', fontsize=11, color="black")

plt.tight_layout()
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300)
plt.show()
