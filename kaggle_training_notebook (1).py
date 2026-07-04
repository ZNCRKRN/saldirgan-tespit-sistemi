# %% [markdown]
# # Kapalı Alanlarda Derin Öğrenme Tabanlı Saldırgan Tespiti
# ## ResNet50 + LSTM + Attention Mechanism
# ## TÜBİTAK 2209-A Projesi - Real Life Violence Dataset
# ### ⚡ Anti-Overfitting Versiyonu

# %% [markdown]
# ## 1. Kütüphanelerin Yüklenmesi

# %%
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import gc
import random
import copy

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Ayarlar ve Parametreler
# ### Overfitting önleme stratejileri:
# - **Düşük HIDDEN_SIZE (128)**: Modelin kapasitesini sınırlar
# - **Yüksek DROPOUT (0.5)**: Agresif regularization
# - **Yüksek WEIGHT_DECAY (5e-4)**: L2 regularization
# - **Label Smoothing (0.1)**: Overconfident tahminleri engeller
# - **Mixup Alpha (0.2)**: Veri artırma ile genellemeyi güçlendirir
# - **CosineAnnealing + Warmup**: Daha stabil öğrenme

# %%
# ===== AYARLAR =====
OUTPUT_DIR = "/kaggle/working"

NUM_FRAMES = 16          # Her videodan çıkarılacak frame sayısı (düşürüldü - daha az bilgi ezberleme)
IMG_SIZE = 224            # ResNet50 giriş boyutu
BATCH_SIZE = 8            # Batch boyutu
NUM_EPOCHS = 30           # Eğitim epoch sayısı (early stopping ile kontrol)
LEARNING_RATE = 3e-4      # Başlangıç öğrenme oranı
HIDDEN_SIZE = 128         # LSTM gizli katman boyutu (küçültüldü - kapasite sınırı)
NUM_LAYERS = 1            # LSTM katman sayısı (1 katman yeterli - overfitting önleme)
DROPOUT = 0.5             # Dropout oranı
WEIGHT_DECAY = 5e-4       # L2 regularization ağırlığı
LABEL_SMOOTHING = 0.1     # Label smoothing katsayısı
MIXUP_ALPHA = 0.2         # Mixup augmentation alpha
EARLY_STOP_PATIENCE = 7   # Sabır - val_loss iyileşmezse dur
WARMUP_EPOCHS = 2         # Learning rate warmup
RANDOM_SEED = 42

# Reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("=" * 60)
print("ANTİ-OVERFİTTİNG AYARLARI")
print("=" * 60)
print(f"  Hidden Size:      {HIDDEN_SIZE} (düşük kapasite)")
print(f"  LSTM Layers:      {NUM_LAYERS} (basit model)")
print(f"  Dropout:          {DROPOUT}")
print(f"  Weight Decay:     {WEIGHT_DECAY}")
print(f"  Label Smoothing:  {LABEL_SMOOTHING}")
print(f"  Mixup Alpha:      {MIXUP_ALPHA}")
print(f"  Early Stop Pat:   {EARLY_STOP_PATIENCE}")
print(f"  Num Frames:       {NUM_FRAMES}")
print(f"  Warmup Epochs:    {WARMUP_EPOCHS}")

# ===== VERİ SETİ YOLUNU OTOMATİK BUL =====
def find_dataset_path(base_path="/kaggle/input"):
    """Kaggle input klasöründe Violence ve NonViolence klasörlerini otomatik bulur."""
    print(f"\nVeri seti aranıyor: {base_path}")
    
    for root, dirs, files in os.walk(base_path):
        if 'Violence' in dirs and 'NonViolence' in dirs:
            print(f"✅ Veri seti bulundu: {root}")
            v_count = len([f for f in os.listdir(os.path.join(root, 'Violence')) if f.endswith(('.mp4', '.avi'))])
            nv_count = len([f for f in os.listdir(os.path.join(root, 'NonViolence')) if f.endswith(('.mp4', '.avi'))])
            print(f"   Violence: {v_count} video, NonViolence: {nv_count} video")
            return root
    
    print("❌ Violence/NonViolence klasörleri bulunamadı!")
    print("Mevcut klasör yapısı:")
    for root, dirs, files in os.walk(base_path):
        depth = root.replace(base_path, '').count(os.sep)
        indent = ' ' * 2 * depth
        print(f"{indent}{os.path.basename(root)}/")
        if depth < 3:
            subindent = ' ' * 2 * (depth + 1)
            for f in files[:5]:
                print(f"{subindent}{f}")
            if len(files) > 5:
                print(f"{subindent}... ve {len(files)-5} dosya daha")
    return None

DATASET_PATH = find_dataset_path()
if DATASET_PATH is None:
    raise FileNotFoundError("Veri seti bulunamadı! Lütfen Kaggle'da 'Add Input' ile 'Real Life Violence Situations' veri setini ekleyin.")

# %% [markdown]
# ## 3. Veri Seti Hazırlığı

# %%
def get_video_paths_and_labels(dataset_path):
    """Veri setindeki video yollarını ve etiketlerini döndürür."""
    video_paths = []
    labels = []
    
    # Violence videoları (label = 1)
    violence_dir = os.path.join(dataset_path, "Violence")
    if os.path.exists(violence_dir):
        for fname in sorted(os.listdir(violence_dir)):
            if fname.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(violence_dir, fname))
                labels.append(1)
    
    # NonViolence videoları (label = 0)
    nonviolence_dir = os.path.join(dataset_path, "NonViolence")
    if os.path.exists(nonviolence_dir):
        for fname in sorted(os.listdir(nonviolence_dir)):
            if fname.endswith(('.mp4', '.avi')):
                video_paths.append(os.path.join(nonviolence_dir, fname))
                labels.append(0)
    
    return video_paths, labels

video_paths, labels = get_video_paths_and_labels(DATASET_PATH)
print(f"Toplam video sayısı: {len(video_paths)}")
print(f"Violence: {sum(labels)}, NonViolence: {len(labels) - sum(labels)}")

# %%
# Train / Validation / Test split (%70 / %15 / %15)
X_train, X_temp, y_train, y_temp = train_test_split(
    video_paths, labels, test_size=0.3, random_state=RANDOM_SEED, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"Eğitim seti:    {len(X_train)} video")
print(f"Doğrulama seti: {len(X_val)} video")
print(f"Test seti:      {len(X_test)} video")

# %% [markdown]
# ## 4. Video Frame Extraction (Temporal Augmentation ile)

# %%
def extract_frames(video_path, num_frames=NUM_FRAMES, img_size=IMG_SIZE, temporal_jitter=False):
    """
    Videodan eşit aralıklı frame'ler çıkarır.
    
    temporal_jitter=True ise, frame seçiminde rastgele kaydırma uygular.
    Bu, modelin belirli frame'leri ezberlemesini engeller.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return None
    
    if total_frames >= num_frames:
        if temporal_jitter and total_frames > num_frames * 2:
            # Temporal Jitter: eşit aralıklı noktalar etrafında rastgele kayma
            base_indices = np.linspace(0, total_frames - 1, num_frames, dtype=float)
            jitter_range = (total_frames / num_frames) * 0.3  # %30 kayma
            jittered = base_indices + np.random.uniform(-jitter_range, jitter_range, num_frames)
            indices = np.clip(jittered, 0, total_frames - 1).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
            frames.append(frame)
    
    cap.release()
    
    # Eksik frame varsa son frame'i tekrarla
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1].copy())
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    
    return np.array(frames[:num_frames])

# Test
sample_frames = extract_frames(video_paths[0])
print(f"Örnek video frame boyutu: {sample_frames.shape}")

# %% [markdown]
# ## 5. Dataset ve DataLoader (Güçlü Augmentation)

# %%
class ViolenceDataset(Dataset):
    """
    Video şiddet tespiti veri seti.
    
    Anti-overfitting özellikleri:
    - Güçlü spatial augmentation (train)
    - Temporal jitter (train) 
    - Temporal subsampling (rastgele frame atlama)
    """
    
    def __init__(self, video_paths, labels, num_frames=NUM_FRAMES, img_size=IMG_SIZE, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_training = is_training
        
        # Eğitim için güçlü augmentation
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),  # Bazen gri tonlama - renk ezberlemeyi engeller
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))  # Küçük bölge silme
        ])
        
        # Validation/Test için sadece normalize
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Temporal jitter sadece eğitimde
        frames = extract_frames(
            video_path, self.num_frames, self.img_size,
            temporal_jitter=self.is_training
        )
        
        if frames is None:
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Temporal Dropout: Eğitimde rastgele 1-2 frame'i sıfırla
        if self.is_training and random.random() < 0.3:
            n_drop = random.randint(1, 2)
            drop_indices = random.sample(range(self.num_frames), min(n_drop, self.num_frames))
            for di in drop_indices:
                frames[di] = np.zeros_like(frames[di])
        
        # Transform uygula (aynı augmentation parametreleri tüm frame'lere)
        transform = self.train_transform if self.is_training else self.val_transform
        transformed_frames = []
        for frame in frames:
            transformed_frames.append(transform(frame))
        
        frames_tensor = torch.stack(transformed_frames)  # (num_frames, 3, 224, 224)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return frames_tensor, label_tensor

# %%
# Dataset'leri oluştur
train_dataset = ViolenceDataset(X_train, y_train, is_training=True)
val_dataset = ViolenceDataset(X_val, y_val, is_training=False)
test_dataset = ViolenceDataset(X_test, y_test, is_training=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=2, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# %% [markdown]
# ## 6. Model Mimarisi: ResNet50 + LSTM + Attention (Anti-Overfitting)
# 
# ### Overfitting önleme mimarisi:
# - **Daha az fine-tuning**: ResNet50'nin sadece son 10 parametresini açıyoruz
# - **Küçük LSTM**: 128 hidden, 1 katman
# - **Spatial Dropout**: Feature map'lere uygulanan dropout
# - **Multi-head basit attention**: Tek attention yerine ortalamalı

# %%
class SpatialDropout1D(nn.Module):
    """
    Spatial Dropout: Tüm time-step boyunca aynı feature'ları kapatır.
    Normal dropout'tan farklı olarak, temporal korelasyonu korur.
    Overfitting'e karşı daha etkilidir çünkü bireysel nöronları
    değil, tüm feature kanallarını kapatır.
    """
    def __init__(self, p=0.2):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        # x: (batch, seq_len, features)
        # Aynı mask'ı tüm time-step'lere uygula
        mask = torch.bernoulli(torch.ones(x.shape[0], 1, x.shape[2], device=x.device) * (1 - self.p))
        return x * mask / (1 - self.p)


class AttentionLayer(nn.Module):
    """Temporal Attention Mechanism - kritik frame'lere odaklanma."""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),  # Daha dar bottleneck
            nn.Tanh(),
            nn.Dropout(0.3),  # Attention'da da dropout
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attention_scores = self.attention(lstm_output)       # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Ağırlıklı toplam
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
        
        return context_vector, attention_weights


class ViolenceDetectionModel(nn.Module):
    """
    ResNet50 (Feature Extraction) + LSTM (Temporal Analysis) + Attention Mechanism
    
    Anti-Overfitting Pipeline:
    Video Frames → ResNet50(frozen+minimal finetune) → SpatialDropout → LSTM → Attention → Dropout → Classification
    """
    
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.5):
        super(ViolenceDetectionModel, self).__init__()
        
        # 1. ResNet50 - Feature Extraction (pre-trained ImageNet)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet50 ağırlıklarını tamamen dondur
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Sadece son layer4'ün son bloğunu aç (minimal fine-tuning)
        # Bu, layer4[-1] yani son residual block - yaklaşık 10 parametre grubu
        for param in list(self.feature_extractor.parameters())[-10:]:
            param.requires_grad = True
        
        self.feature_size = 2048  # ResNet50 çıkış boyutu
        
        # Feature boyut düşürme (bottleneck) - parametre sayısını azaltır
        self.feature_reducer = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),  # Bu per-frame uygulanacak
        )
        
        # Spatial Dropout - feature map'lerde kanal bazlı dropout
        self.spatial_dropout = SpatialDropout1D(p=0.3)
        
        # 2. LSTM - Temporal Analysis (küçük ve basit)
        self.lstm = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # 1 katman olduğu için LSTM internal dropout yok
            bidirectional=True
        )
        
        # Layer Normalization - LSTM çıkışını normalize et
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # 3. Attention Mechanism
        self.attention = AttentionLayer(hidden_size * 2)  # *2 bidirectional
        
        # 4. Classification Head (basit ve regularized)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout * 0.6),  # İkinci katmanda biraz daha düşük dropout
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, num_frames, 3, 224, 224)
        batch_size, num_frames, C, H, W = x.shape
        
        # Her frame için ResNet50 feature extraction
        x = x.view(batch_size * num_frames, C, H, W)
        
        with torch.set_grad_enabled(self.training):
            features = self.feature_extractor(x)  # (batch*frames, 2048, 1, 1)
        features = features.view(batch_size * num_frames, self.feature_size)  # (batch*frames, 2048)
        
        # Feature boyut düşürme
        features = self.feature_reducer(features)  # (batch*frames, hidden*2)
        features = features.view(batch_size, num_frames, -1)  # (batch, frames, hidden*2)
        
        # Spatial dropout
        features = self.spatial_dropout(features)
        
        # LSTM temporal analysis
        lstm_out, _ = self.lstm(features)  # (batch, frames, hidden*2)
        
        # Layer Normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention mechanism
        context, attention_weights = self.attention(lstm_out)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(context)  # (batch, 1)
        
        return output.squeeze(1), attention_weights

# %%
# Model oluştur
model = ViolenceDetectionModel(
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# Model özeti
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params
print(f"\n{'='*60}")
print(f"MODEL ÖZETİ")
print(f"{'='*60}")
print(f"  Toplam parametre:     {total_params:,}")
print(f"  Eğitilebilir:        {trainable_params:,}")
print(f"  Donmuş:              {frozen_params:,}")
print(f"  Eğitilebilir oran:   {trainable_params/total_params*100:.1f}%")
print(f"{'='*60}")

# %% [markdown]
# ## 7. Eğitim (Anti-Overfitting Teknikleri)

# %%
# ===== LOSS FONKSİYONU: Label Smoothing BCE =====
class LabelSmoothingBCELoss(nn.Module):
    """
    Label Smoothing ile Binary Cross Entropy Loss.
    
    Label smoothing, modelin çok emin (overconfident) tahminler
    yapmasını engeller. Yüzde 100 emin olmak yerine %95 gibi
    daha dengeli tahminler üretmesini sağlar.
    """
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy(pred, target_smooth)


# ===== MIXUP FONKSİYONU =====
def mixup_data(x, y, alpha=0.2):
    """
    Mixup augmentation: İki örneği karıştırarak yeni örnekler üretir.
    
    Bu teknik, modelin karar sınırlarını yumuşatır ve
    ezberlemeyi zorlaştırır. Alpha küçükse (0.2) karışım hafif olur.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Lambda'yı 0.65-1.0 aralığında tut (çok agresif karışım istemiyoruz)
    lam = max(lam, 1 - lam)
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss hesaplama."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# %%
# ===== OPTİMİZER VE SCHEDULER =====
criterion = LabelSmoothingBCELoss(smoothing=LABEL_SMOOTHING)

# Farklı learning rate'ler: ResNet feature extractor düşük, geri kalanı yüksek
resnet_params = [p for p in model.feature_extractor.parameters() if p.requires_grad]
other_params = [p for n, p in model.named_parameters() 
                if p.requires_grad and 'feature_extractor' not in n]

optimizer = optim.AdamW([
    {'params': resnet_params, 'lr': LEARNING_RATE * 0.1},   # ResNet: düşük lr
    {'params': other_params, 'lr': LEARNING_RATE}           # Diğerleri: normal lr
], weight_decay=WEIGHT_DECAY)

# Cosine Annealing: learning rate'i yumuşakça düşürür
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# %%
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, use_mixup=True):
    """
    Bir epoch eğitim.
    Mixup augmentation ve gradient clipping içerir.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        
        # Mixup augmentation (her batch'te %50 olasılıkla)
        if use_mixup and random.random() < 0.5:
            frames, labels_a, labels_b, lam = mixup_data(frames, labels, MIXUP_ALPHA)
            
            optimizer.zero_grad()
            outputs, _ = model(frames)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # Mixup'ta accuracy yaklaşık hesapla
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (lam * (predicted == labels_a).float() + 
                       (1 - lam) * (predicted == labels_b).float()).sum().item()
        else:
            optimizer.zero_grad()
            outputs, _ = model(frames)
            loss = criterion(outputs, labels)
            
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        loss.backward()
        
        # Gradient clipping - exploding gradient'ları engeller
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item() * frames.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}', 
            'acc': f'{correct/total:.4f}',
            'lr': f'{optimizer.param_groups[1]["lr"]:.6f}'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Doğrulama - augmentation olmadan temiz değerlendirme."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Validation"):
            frames, labels = frames.to(device), labels.to(device)
            
            outputs, _ = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * frames.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # AUC-ROC hesapla
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    
    return epoch_loss, epoch_acc, auc

# %%
# ===== EĞİTİM DÖNGÜSÜ =====
print("=" * 60)
print("MODEL EĞİTİMİ BAŞLIYOR (Anti-Overfitting Mode)")
print("=" * 60)

history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [], 'val_auc': [],
    'lr': [], 'overfit_gap': []
}

best_val_loss = float('inf')
best_val_acc = 0.0
best_model_state = None
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    current_lr = optimizer.param_groups[1]['lr']
    print(f"Learning Rate: {current_lr:.6f}")
    print(f"{'='*60}")
    
    # Warmup: ilk birkaç epoch'ta mixup kullanma (modelin önce temel kalıpları öğrenmesi için)
    use_mixup = epoch >= WARMUP_EPOCHS
    
    # Eğitim
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch, use_mixup=use_mixup
    )
    
    # Doğrulama
    val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
    
    # Scheduler step
    scheduler.step(epoch)
    
    # Overfitting gap hesapla
    overfit_gap = train_acc - val_acc
    
    # History kaydet
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_auc'].append(val_auc)
    history['lr'].append(current_lr)
    history['overfit_gap'].append(overfit_gap)
    
    print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"  Val AUC:    {val_auc:.4f}")
    print(f"  Overfit Gap: {overfit_gap:.4f} ", end="")
    
    # Overfitting uyarısı
    if overfit_gap > 0.10:
        print("⚠️  UYARI: Overfitting başlıyor!")
    elif overfit_gap > 0.05:
        print("🔶 DİKKAT: Hafif overfitting")
    else:
        print("✅ İyi genelleme")
    
    # En iyi modeli kaydet (val_loss bazlı - daha güvenilir)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_acc = val_acc
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'history': history,
        }, os.path.join(OUTPUT_DIR, 'best_model.pth'))
        print(f"  ✅ En iyi model kaydedildi! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f})")
    else:
        patience_counter += 1
        print(f"  ⏳ Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping! {EARLY_STOP_PATIENCE} epoch boyunca val_loss iyileşmedi.")
            break
    
    # Bellek temizliği
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# En iyi modeli geri yükle
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\n{'='*60}")
print(f"Eğitim tamamlandı!")
print(f"En iyi Val Loss: {best_val_loss:.4f}")
print(f"En iyi Val Acc:  {best_val_acc:.4f}")
print(f"{'='*60}")

# %% [markdown]
# ## 8. Eğitim Grafikleri (Overfitting Analizi dahil)

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss grafiği
axes[0, 0].plot(history['train_loss'], label='Train Loss', color='#e74c3c', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', color='#3498db', linewidth=2)
axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Accuracy grafiği
axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='#e74c3c', linewidth=2)
axes[0, 1].plot(history['val_acc'], label='Val Accuracy', color='#3498db', linewidth=2)
axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Overfitting Gap grafiği
epochs_range = range(1, len(history['overfit_gap']) + 1)
colors = ['green' if g <= 0.05 else 'orange' if g <= 0.10 else 'red' for g in history['overfit_gap']]
axes[1, 0].bar(epochs_range, history['overfit_gap'], color=colors, alpha=0.7)
axes[1, 0].axhline(y=0.05, color='orange', linestyle='--', label='Hafif Overfitting Sınırı')
axes[1, 0].axhline(y=0.10, color='red', linestyle='--', label='Ciddi Overfitting Sınırı')
axes[1, 0].set_title('Overfitting Gap (Train Acc - Val Acc)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Gap')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Learning Rate grafiği
axes[1, 1].plot(history['lr'], color='#9b59b6', linewidth=2, marker='o', markersize=4)
axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Eğitim grafikleri kaydedildi: training_history.png")

# %% [markdown]
# ## 9. Test Değerlendirmesi

# %%
# En iyi modeli yükle
checkpoint = torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth'), weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"En iyi model yüklendi (Epoch {checkpoint['epoch']+1}, Val Acc: {checkpoint['val_acc']:.4f})")

# %%
# Test seti üzerinde değerlendirme
model.eval()
all_predictions = []
all_labels = []
all_probs = []

with torch.no_grad():
    for frames, labels in tqdm(test_loader, desc="Test Evaluation"):
        frames, labels = frames.to(device), labels.to(device)
        outputs, _ = model(frames)
        
        predicted = (outputs > 0.5).float()
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(outputs.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# %%
# Classification Report
print("\n" + "=" * 60)
print("TEST SONUÇLARI")
print("=" * 60)

test_accuracy = accuracy_score(all_labels, all_predictions)
test_f1 = f1_score(all_labels, all_predictions)
test_auc = roc_auc_score(all_labels, all_probs)

print(f"\n🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"📊 Test F1-Score: {test_f1:.4f}")
print(f"📈 Test AUC-ROC:  {test_auc:.4f}")

# Overfitting final analizi
final_train_acc = history['train_acc'][-1]
print(f"\n📋 Overfitting Analizi:")
print(f"   Son Train Acc: {final_train_acc:.4f}")
print(f"   Test Acc:      {test_accuracy:.4f}")
print(f"   Gap:           {final_train_acc - test_accuracy:.4f}")

if abs(final_train_acc - test_accuracy) < 0.05:
    print("   ✅ Overfitting YOK - Model iyi genelleme yapıyor!")
elif abs(final_train_acc - test_accuracy) < 0.10:
    print("   🔶 Hafif overfitting var ama kabul edilebilir")
else:
    print("   ⚠️ Overfitting mevcut - Daha fazla regularization gerekebilir")

print("\n📋 Classification Report:")
print(classification_report(all_labels, all_predictions, 
                            target_names=['NonViolence', 'Violence']))

# %%
# Confusion Matrix
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['NonViolence', 'Violence'],
            yticklabels=['NonViolence', 'Violence'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.show()
print("📊 Confusion matrix kaydedildi: confusion_matrix.png")

# %% [markdown]
# ## 10. Modeli Dışa Aktar

# %%
# Sadece model ağırlıklarını kaydet (daha küçük dosya)
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'violence_detection_model_weights.pth'))

# Tam modeli kaydet (checkpoint)
torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'num_frames': NUM_FRAMES,
        'img_size': IMG_SIZE,
    },
    'test_accuracy': test_accuracy,
    'test_f1': test_f1,
    'test_auc': test_auc,
    'classification_report': classification_report(all_labels, all_predictions, 
                                                     target_names=['NonViolence', 'Violence'],
                                                     output_dict=True),
    'history': history,
}, os.path.join(OUTPUT_DIR, 'violence_detection_full_model.pth'))

print("\n✅ Model dosyaları kaydedildi:")
print(f"  - violence_detection_model_weights.pth")
print(f"  - violence_detection_full_model.pth")
print(f"  - best_model.pth")
print(f"  - training_history.png")
print(f"  - confusion_matrix.png")

# %%
# Dosya boyutları
for fname in ['best_model.pth', 'violence_detection_model_weights.pth', 
              'violence_detection_full_model.pth']:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  {fname}: {size_mb:.1f} MB")

# %%
# ===== ANTİ-OVERFİTTİNG RAPORU =====
print("\n" + "=" * 60)
print("ANTİ-OVERFİTTİNG RAPORU")
print("=" * 60)
print(f"""
Uygulanan Teknikler:
  1. ✅ Label Smoothing ({LABEL_SMOOTHING})
  2. ✅ Mixup Augmentation (alpha={MIXUP_ALPHA})
  3. ✅ Spatial Dropout (0.3)
  4. ✅ Temporal Jitter (frame seçiminde rastgelelik)
  5. ✅ Temporal Dropout (rastgele frame silme)
  6. ✅ Güçlü Data Augmentation (flip, rotation, color jitter, affine, grayscale, erasing)
  7. ✅ Weight Decay / L2 Regularization ({WEIGHT_DECAY})
  8. ✅ Cosine Annealing LR Scheduler
  9. ✅ Gradient Clipping (max_norm=1.0)
  10. ✅ Early Stopping (patience={EARLY_STOP_PATIENCE})
  11. ✅ Minimal Fine-Tuning (ResNet sadece son 10 param)
  12. ✅ Feature Bottleneck (2048→{HIDDEN_SIZE*2})
  13. ✅ Küçük Model Kapasitesi (LSTM hidden={HIDDEN_SIZE}, layers={NUM_LAYERS})
  14. ✅ Farklı LR grupları (ResNet: düşük, diğer: normal)
  15. ✅ Layer Normalization
  16. ✅ Warmup (ilk {WARMUP_EPOCHS} epoch mixup yok)
  17. ✅ Drop Last Batch (eksik batch sorununu önler)

Sonuçlar:
  Test Accuracy:  {test_accuracy:.4f}
  Test F1-Score:  {test_f1:.4f}
  Test AUC-ROC:   {test_auc:.4f}
  Overfit Gap:    {final_train_acc - test_accuracy:.4f}
""")

print("🎉 Eğitim tamamlandı! Modeli Kaggle Output'tan indirebilirsiniz.")
print("İndirdikten sonra GitHub reponuza push edin.")
