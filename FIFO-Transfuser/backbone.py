import torch
import torch.nn as nn

from refinenetlw import rf_lw101
from transfuser import TransfuserBackbone


# convert module for setting the output of rf_lw101 to the image input of the transfuser
class RF_LW101_To_TransfuserInput(nn.Module):
    def __init__(self, in_channels, target_channels=3, target_size=(224, 224)):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, target_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=target_size, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        return self.upsample(self.relu(self.conv1x1(x)))


# image: rf_lw101 + transfuser, radar: transfuser only
class IntegratedModel(nn.Module):
    def __init__(self, transfuser_params, rf_lw101_params):
        super().__init__()
        
        self.rf_lw101 = rf_lw101(**rf_lw101_params) # only one param: num_classes=19
        
        # load pretrained parameter of rf_lw101
        checkpoint = torch.load('./FIFO_final_model.pth', map_location=torch.device('cpu'))
        self.rf_lw101.load_state_dict(checkpoint['state_dict'])
        #self.rf_lw101.load_state_dict(checkpoint)

        # freeze all params of rf_lw101 (not trained)
        for param in self.rf_lw101.parameters():
            param.requires_grad = False

        self.transfuser = TransfuserBackbone(**transfuser_params)
        
        self.rf_to_transfuser = RF_LW101_To_TransfuserInput(in_channels=19, target_channels=3, target_size=(224, 224)) # check in_channels!
    
    def forward(self, image, radar):
        rf_output1, rf_output2, rf_output3, rf_output4, rf_output5, rf_output6 = self.rf_lw101(image)
        #print(rf_output1.shape, rf_output2.shape, rf_output3.shape, rf_output4.shape, rf_output5.shape, rf_output6.shape)
        
        transformed_image = self.rf_to_transfuser(rf_output6)  # [B, 19, 56, 56]
        
        fused_features = self.transfuser(transformed_image, radar)
        
        return fused_features
    

##### model initialize and test #####


transfuser_params = {
    'img_vert_anchors': 10,
    'img_horz_anchors': 10,
    'radar_vert_anchors': 8,
    'radar_horz_anchors': 8,
    'perception_output_features': 256,
    'use_point_pillars': False,
    'num_features': [64, 128, 256, 512],  # ResNet34의 각 레이어 출력 채널과 일치해야 함
    'radar_seq_len': 1,
    'use_target_point_image': False,
    'n_head': 8,
    'block_exp': 4,
    'n_layer': 2,
    'seq_len': 1,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'heatmap_features_channels': 1,
    'image_architecture': 'resnet34',  # ImageCNN은 기존 이미지 인코더를 유지
    'radar_architecture': 'resnet18',
    'gpt_linear_layer_init_mean': 0.0,
    'gpt_linear_layer_init_std': 0.02,
    'gpt_layer_norm_init_weight': 1.0
}

rf_lw101_params = {
    'num_classes': 19,
}

model = IntegratedModel(transfuser_params, rf_lw101_params)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# generate dummy image data and dummy radar data
dummy_image = torch.randn(2, 3, 224, 224).to(device)
dummy_radar = torch.randn(2, 2, 224, 224).to(device)

with torch.no_grad():
    output = model(dummy_image, dummy_radar)
print("output tensor shape:", output.shape) # [B, 256]

from torchsummary import summary
summary(model, [(3, 224, 224), (2, 224, 224)])