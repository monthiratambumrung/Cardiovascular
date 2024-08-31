import streamlit as st 
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title 
st.title('Cardiovascular Classification')

# Set Header 
st.header('Please upload a picture')

# Load Model 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('mobilenetv3_large_100_checkpoint_fold1.pt', map_location=device)
model.to(device)

# Display image & Prediction 
uploaded_image = st.file_uploader('Choose an image', type=['jpg'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_name = ['true', 'false']

    if st.button('Prediction'):
        try:
            # Prediction class
            pred, probli = pred_class(model, image, class_name)
            
            st.write("## Prediction Result")
            max_index = np.argmax(probli[0])

            for i in range(len(class_name)):
                color = "blue" if i == max_index else None
                st.write(f"## <span style='color:{color}'>{class_name[i]} : {probli[0][i]*100:.2f}%</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error occurred: {e}")
