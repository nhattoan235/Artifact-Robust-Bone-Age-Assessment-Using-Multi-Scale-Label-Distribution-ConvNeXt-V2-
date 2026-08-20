from PIL import Image
import matplotlib.pyplot as plt
import os

original_path = r"C:\Projetos\hand-bone\boneage-test-dataset\boneage-test-dataset\4362.png"
cleaned_path  = r"C:\Projetos\hand-bone\output_cleaned_mult\4362\4362_cleaned_1.png"
save_path     = r"C:\Projetos\hand-bone\figs\comparison_4362.png"

img_orig = Image.open(original_path)
img_clean = Image.open(cleaned_path)

fig, axes = plt.subplots(1, 2, figsize=(10, 6))

axes[0].imshow(img_orig, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(img_clean, cmap='gray')
axes[1].set_title("Inpainted Image")
axes[1].axis("off")

plt.tight_layout(rect=[0, 0.05, 1, 1])

os.makedirs(os.path.dirname(save_path), exist_ok=True)
fig.savefig(save_path, dpi=300)

plt.show()
