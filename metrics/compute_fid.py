"""
About:
Файл для подсчёта метрики FID (Fréchet Inception Distance). Вместе с файлом, в корне должны находиться папки image, GAN, LADI, PROMPT, где 

image - исходные изображения
GAN - выходы модели PASTA-GAN++
LADI - выходы модели LaDI-VITON
PROMPT - выходы модели PromptDresser

Результирующий barplot будет сохранён в корне, с названием fid_results.png
"""
import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from scipy import linalg
import matplotlib.pyplot as plt

def load_and_preprocess_matched(source_folder: str, target_folder: str):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # Словари: имя_без_расширения -> имя_файла
    source_files = {os.path.splitext(f)[0]: f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png'))}
    target_files = {os.path.splitext(f)[0]: f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.png'))}
    # Пересечение по имени
    common_keys = sorted(set(source_files.keys()) & set(target_files.keys()))
    src_imgs, tgt_imgs = [], []
    for key in common_keys:
        src_img = Image.open(os.path.join(source_folder, source_files[key])).convert('RGB')
        tgt_img = Image.open(os.path.join(target_folder, target_files[key])).convert('RGB')
        src_imgs.append(transform(src_img))
        tgt_imgs.append(transform(tgt_img))

    if src_imgs and tgt_imgs:
        return torch.stack(src_imgs), torch.stack(tgt_imgs)
    else:
        return torch.tensor([]), torch.tensor([])

def extract_feats(tensor, model, device):
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(tensor), 32):
            batch = tensor[i:i+32].to(device)
            feats.append(model(batch).cpu().numpy())
    return np.vstack(feats)

def stats(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def fid_score(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

# Вычисление FID
def get_fid(src_imgs, tgt_imgs):
    if src_imgs.nelement() == 0 or tgt_imgs.nelement() == 0:
        return None
    feats_src = extract_feats(src_imgs, model, device)
    feats_tgt = extract_feats(tgt_imgs, model, device)
    mu_s, sig_s = stats(feats_src)
    mu_t, sig_t = stats(feats_tgt)
    return fid_score(mu_s, sig_s, mu_t, sig_t)

# Пути к папкам
source = "image/"
gan = "GAN/"
ladi = "LADI/"
prompt = "PROMPT/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = inception_v3(pretrained=True)
model.fc = torch.nn.Identity()
model.to(device)

#Загрузка файлов
src_gan_src, src_gan_tgt = load_and_preprocess_matched(source, gan)
src_ladi_src, src_ladi_tgt = load_and_preprocess_matched(source, ladi)
src_prm_src, src_prm_tgt = load_and_preprocess_matched(source, prompt)



fid_gan = get_fid(src_gan_src, src_gan_tgt)
fid_ladi = get_fid(src_ladi_src, src_ladi_tgt)
fid_prm = get_fid(src_prm_src, src_prm_tgt)

#Вывод результатов
models = []
values = []
if fid_gan is not None:
    models.append("PASTA-GAN++")
    values.append(fid_gan)
if fid_ladi is not None:
    models.append("LADI-VITON")
    values.append(fid_ladi)
if fid_prm is not None:
    models.append("PromptDresser")
    values.append(fid_prm)

for m, v in zip(models, values):
    print(f"FID для {m}: {v:.2f}")

#Отображение графика, сохранение
plt.figure(figsize=(8, 5))
bars = plt.bar(models, values, color=['skyblue', 'lightgreen', 'salmon'][:len(values)])
plt.ylabel("FID")
plt.title("Результаты подсчёта FID для моделей")

for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, value + 0.5, f"{value:.2f}", ha='center')

plt.ylim(0, max(values)*1.1)
plt.show()
plt.savefig('fid_results.png')
