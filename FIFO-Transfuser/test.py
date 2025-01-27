import torch
from refinenetlw import rf_lw101
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 이미지 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 모델의 입력 크기에 맞게 이미지 리사이즈
        transforms.ToTensor(),          # 이미지를 텐서로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)  # 배치 차원 추가

# 모델의 추론을 수행하는 함수
def test(image_path):
    # 1. 모델 초기화
    model = rf_lw101(num_classes=19)
    
    # 2. 가중치 불러오기 (필요한 부분만 추출)
    #checkpoint = torch.load('./Cityscapes_pretrained_model.pth', map_location=torch.device('cpu'))
    checkpoint = torch.load('./FIFO_final_model.pth', map_location=torch.device('cpu'))
    
    # 모델 가중치만 추출하여 로드
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)  # 직접 state_dict가 아닐 경우 전체 로드 시도

    # 3. 모델을 평가 모드로 설정
    model.eval()
    
    # 4. 입력 이미지 전처리
    input_image = preprocess_image(image_path)
    
    # 5. 추론 수행
    with torch.no_grad():
        output = model(input_image)
    
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
    print(output[4].shape)
    print(output[5].shape)

    # 6. output이 tuple인 경우 첫 번째 요소 선택
    if isinstance(output, tuple):
        output = output[5] # out6 of refinenetlw
    
    # 7. 결과 후처리 (여기서는 가장 높은 클래스를 선택하여 segmentation map 생성)
    prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    return prediction

# Segmentation 결과 시각화 함수
def visualize_segmentation(image_path, prediction):
    # 원본 이미지 불러오기
    original_image = Image.open(image_path).convert('RGB')
    
    # Prediction 시각화 (정수를 컬러맵으로 변환)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap='jet')  # 예측 결과를 컬러맵으로 표시
    plt.title('Segmentation Result')
    plt.axis('off')

    plt.show()

# 예시 실행
if __name__ == '__main__':
    image_path = './test.jpg'  # 테스트할 이미지 경로
    result = test(image_path)
    
    # 결과 시각화
    visualize_segmentation(image_path, result)