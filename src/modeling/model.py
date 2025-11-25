import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
#    3D Deep Convolutional Neural Network
# ==========================================
class Drowsiness3DCNN(nn.Module):
    def __init__(self):
        super(Drowsiness3DCNN, self).__init__()
        
        # input: (batch, 3, 5, 224, 224)
        # block 1: 7 x 7 x 2 kernel
        self.conv1 = nn.Conv3d(3, 128, kernel_size=(2, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.bn1 = nn.BatchNorm3d(128)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # block 2: 1x1 Kernel (Bottleneck)
        self.conv2 = nn.Conv3d(128, 128, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # block 3: 3x3 Kernel
        self.conv3 = nn.Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)
        
        # block 4: 1x1 Kernel (Depth Increase)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=(1, 1, 1), stride=1, padding=0)
        self.bn4 = nn.BatchNorm3d(256)
        
        # block 5: 3x3 Kernel
        self.conv5 = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn5 = nn.BatchNorm3d(256)
        
        # block 6: 3x3 Kernel
        self.conv6 = nn.Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.bn6 = nn.BatchNorm3d(256)
        
        # hitung ukuran output conv secara otomatis untuk Flatten
        self._to_linear = None
        self._get_conv_output((3, 5, 224, 224))
        
        # fully Connected untuk menghasilkan Alpha (512)
        self.fc_alpha = nn.Linear(self._to_linear, 512)

    # fungsi bantuan untuk menghitung ukuran output conv secara otomatis
    # ini penting agar tidak error "shape mismatch" di Linear layer
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self._forward_features(input)
            self._to_linear = int(output.view(1, -1).size(1))

    def _forward_features(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        # Block 4
        x = F.relu(self.bn4(self.conv4(x)))
        # Block 5
        x = F.relu(self.bn5(self.conv5(x)))
        # Block 6
        x = F.relu(self.bn6(self.conv6(x)))
        
        return x

    def forward(self, x):
        # ekstraksi fitur
        x = self._forward_features(x)
        
        # flattening (meratakan Tensor menjadi vektor)
        x = x.view(x.size(0), -1) # Flatten

        # menghasilkan Alpha (512)
        alpha = self.fc_alpha(x)  # Spatio-temporal representation
        
        return alpha

# ==========================================
#         Scene Understanding Model
# ==========================================
class SceneUnderstanding(nn.Module):
    def __init__(self):
        super(SceneUnderstanding, self).__init__()

        # input: 512 (Alpha dari 3D-DCNN)
        # layer 1: fully connected network (512)
        self.fc1 = nn.Linear(in_features=512, out_features=512)

        # Layer 2: fully connected network (256)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        
        # output layer: 13 neuron
        # rincian 13 output:
        # 1. glasses (5)
        # 2. head (3)
        # 3. mouth (3)
        # 4. eye (2)
        self.fc_out = nn.Linear(in_features=256, out_features=13) 

        self.relu = nn.ReLU()

    def forward(self, alpha):
        """
        args:
            alpha: Tensor input dari 3D-DCNN dengan ukuran (Batch_Size, 512)
        Returns:
            pred_labels: Tensor output (Lc) dengan ukuran (Batch_Size, 13)
        """

        # proses layer 1
        x = self.relu(self.fc1(alpha))
        
        # proses layer 2
        x = self.relu(self.fc2(x))
        
        # output akhir (Lc)
        # note: kita tidak pakai softmax di sini karena ini output mentah (logits).
        # softmax nanti dilakukan saat perhitungan loss atau di Fusion Model.
        pred_labels = self.fc_out(x) # Lc (Logits)
        
        return pred_labels

# ==========================================
#               Fusion Model
# ==========================================
class FusionModel(nn.Module):
    def __init__(self, alpha_dim=512, label_dim=13, hidden_dim=512, output_dim=512):
        super(FusionModel, self).__init__()

        # beta = W_fu ( W_fea * alpha (X) W_sc * Lc ) + b_fu

        # 1. W_fea: projection untuk Alpha (fitur video)
        # input: 512 -> output: 512 (hidden_dim)
        # bias=False karena di rumus hanya W * alpha
        self.W_fea = nn.Linear(alpha_dim, hidden_dim, bias=False)
        
        # 2. W_sc: projection untuk condition labels
        # input: 13 -> output: 512 (hidden_dim) agar bisa dikalikan dengan alpha
        # bias=False karena di rumus hanya W * Lc
        self.W_sc = nn.Linear(label_dim, hidden_dim, bias=False)
        
        # 3. W_fu & b_fu: post-fusion projection
        # input: 512 -> output: 512 (output_dim)
        # bias=True untuk merepresentasikan "+ b_fu"
        self.W_fu = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, alpha, labels):
        """
        args:
            alpha: Tensor (batch, 512) dari 3D-DCNN
            labels: Tensor (batch, 13) dari Scene Understanding Model
        Returns:
            v: Tensor (Batch, 512) - normalized condition-specified features
        """
        
        # --- proyeksi (projection) ---
        # membentuk W_fea * alpha
        proj_alpha = self.W_fea(alpha)   # (batch, 512)
        
        # membentuk W_sc * Lc
        proj_labels = self.W_sc(labels)  # (batch, 512)
        
        # --- multiplicative interaction (simbol X dalam lingkaran) ---
        # element-wise multiplication
        # ini intinya: label kondisi menjadi "masker" bagi fitur video
        fused = proj_alpha * proj_labels # (batch, 512)

        # --- Transformasi Akhir untuk mendapatkan Beta ---
        # beta = W_fu(...) + b_fu
        beta = self.W_fu(fused)
        
        # --- Normalisasi (Rumus 6) ---
        # v = softmax(beta)
        # mengubah fitur menjadi probabilitas/attention weight
        v = F.softmax(beta, dim=1)  # (batch, 512)
        
        return v

# ==========================================
#         Drowsiness Detection Model
# ==========================================
class DrowsinessDetection(nn.Module):
    def __init__(self, input_dim=512):
        super(DrowsinessDetection, self).__init__()
        
        # layer 1: fully connected (512 -> 512)
        self.fc1 = nn.Linear(input_dim, 512)
        
        # layer 2: fully connected (512 -> 256)
        self.fc2 = nn.Linear(512, 256)
        
        # output layer: 2 kategori (drowsy vs non-drowsy)
        self.fc_out = nn.Linear(256, 2)
        
        self.relu = nn.ReLU()

    def forward(self, v):
        """
        args:
            v: Tensor (Batch, 512) - fitur hasil fusion
        returns:
            logits: Tensor (Batch, 2) - skor mentah untuk [non-drowsy, drowsy]
        """
        x = self.relu(self.fc1(v))
        x = self.relu(self.fc2(x))

        # output Logits
        # kita kembalikan logits (nilai mentah), bukan probabilitas.
        # nanti CrossEntropyLoss yang akan menghitung softmax-nya secara internal saat training.
        logits = self.fc_out(x)

        return logits

# ==========================================
#                 MAIN SYSTEM 
# ==========================================
class DrowsinessDetectionSystem(nn.Module):
    def __init__(self):
        super(DrowsinessDetectionSystem, self).__init__()
        
        # inisialisasi 4 komponen utama
        self.dcnn = Drowsiness3DCNN()
        self.scene_model = SceneUnderstanding()
        self.fusion_model = FusionModel()
        self.detection_model = DrowsinessDetection()

    def forward(self, x):
        # 1. ekstrak spatio-temporal representation (alpha)
        alpha = self.dcnn(x)
        
        # 2. prediksi scene condition (Lc)
        scene_logits = self.scene_model(alpha)
        
        
        # 3. buat fusion features (v)
        # fusion mengambil Alpha dan Scene Logits sebagai input
        v = self.fusion_model(alpha, scene_logits)
        
        
        # 4. deteksi kantuk akhir
        drowsiness_logits = self.detection_model(v)
        
        
        # return kedua output agar bisa dihitung loss-nya (scene loss & detection loss)
        return scene_logits, drowsiness_logits