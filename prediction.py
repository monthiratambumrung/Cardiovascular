from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

def pred_class(model: torch.nn.Module,
               image: Image.Image,  # เพิ่ม type hint ให้ชัดเจน
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224)) -> Tuple[str, np.ndarray]:
    
    # แปลงรูปภาพให้เหมาะสมก่อนนำเข้าโมเดล
    image_transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # ปิดลิสต์ที่นี่
                    std=[0.229, 0.224, 0.225]),  # ปิดลิสต์ที่นี่
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    with torch.inference_mode():
        transformed_image = image_transform(image).unsqueeze(dim=0).float()
        transformed_image = transformed_image.to(device)

        target_image_pred = model(transformed_image)
        target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

        # หาคลาสที่มีค่าความน่าจะเป็นสูงสุด
        target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
        predicted_class = class_names[target_image_pred_label.item()]
        prob = target_image_pred_probs.cpu().numpy()

    return predicted_class, prob
