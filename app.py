import streamlit as st
import pickle, os, random
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from rag import Pipeline, DeepSeek
from streamlit_image_select import image_select
from skimage import transform
import uuid

TRAINING_DATA_IMG = r"/Users/revathsankar/Documents/Adv Big AI/Midterm/output.png"
COLUMBIA_LOGO = "https://visualidentity.columbia.edu/sites/default/files/styles/cu_crop/public/private/2023-logo-blue-white.jpg?itok=H8HIEYVa"
tumor_type = random.choice(['glioma', 'meningioma', 'pituitary_tumor','no_tumor' ])
IMAGE_DIR = fr'/Users/revathsankar/Documents/Adv Big AI/Midterm/Brain-MRI-Image-Classification-Using-Deep-Learning/Brain-Tumor-Dataset/Training/{tumor_type}'
image_size = 128
batch_size = 32

def translate_classes(class_no:int):
     class_names = ['Glioma', 'Meningioma', 'No-Tumor', 'Pituitary\nTumor']
     return dict(zip(list(range(0,4)), class_names)).get(class_no)



st.title("Radiologist AI")


@st.cache_resource
def preprocess():
    
    with st.spinner("Loading LLM & Image Analysis Models..."):
        llm = DeepSeek()
        pipeline = Pipeline()
        with open("adv_big_weights.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        sample_image_path = random.choice([os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, x))])
        selectable_images = random.choices([os.path.join(IMAGE_DIR, x) for x in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, x))],k=5)
    return llm, pipeline, model, sample_image_path, selectable_images

llm, pipeline, model, sample_image_path, select_images = preprocess()
with st.form('main-form'):
    with st.sidebar:
        add_uni_logo = st.logo(COLUMBIA_LOGO, size='large')
        submitted = st.form_submit_button("Analyze Image")
        with st.expander("Deep Learning Prediction"):
            st.write("Sample Image for Analysis")
        
            sample_image = st.image(sample_image_path)
            truth_label = st.markdown(f"Actual Tumor Type: {tumor_type}")
        
            if submitted:
                with st.spinner("Loading analysis..."):
                    # image_datagen.flow_from_directory(sample_image_path,
                    #                                   seed = 42, 
                    #                                     batch_size = batch_size,
                    #                                     target_size = (image_size, image_size),
                    #                                     color_mode = 'rgb')
                    img = image.load_img(sample_image_path)
                    img_arr = image.img_to_array(img)
                    resized = transform.resize(img_arr, (128,128,3))
                    expanded_img = np.expand_dims(resized, axis=0)
                    preprocessed_img = preprocess_input(expanded_img)
                    prediction = model.predict(preprocessed_img)
                    st.write(f"Predicted Class: {translate_classes(np.argmax(prediction))}")
                    st.write((prediction))
                    llm_response = llm.analyze_image(llm.get_initial_analysis(img_path=sample_image_path))
                    st.write(f"Image Analysis: {llm_response}")
        st.divider()
        add_image = st.image(TRAINING_DATA_IMG, caption="Training Error", width=250)
        add_uploader = st.file_uploader(label="Upload image of interest to determine blastoma classification.", type=['png','jpg'])
        if add_uploader:
            bytes_data = add_uploader.getvalue()
            st.write(bytes_data)
        img_choices = image_select("Select image to analyze",select_images)

if "messages" not in st.session_state:
        st.session_state.messages = []
else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
# load_sidebar()
if prompt := st.chat_input("Enter your question about the image of interest!"):
    with st.spinner("Synthesizing response.."):
        context = pipeline.retrieve_context(prompt)
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = llm.analyze_image(llm.extend_analysis(query=prompt, context=context, img_path=img_choices))
        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})