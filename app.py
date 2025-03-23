import streamlit as st
import pickle, os, random, zipfile
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from rag import Pipeline, DeepSeek
from streamlit_card import card
from streamlit_image_select import image_select
from skimage import transform
import uuid

TRAINING_DATA_IMG = r"./output.png"
COLUMBIA_LOGO = "https://visualidentity.columbia.edu/sites/default/files/styles/cu_crop/public/private/2023-logo-blue-white.jpg?itok=H8HIEYVa"
tumor_type = random.choice(['glioma', 'meningioma', 'pituitary_tumor','no_tumor' ])
IMAGE_DIR = fr'./Brain-MRI-Image-Classification-Using-Deep-Learning/Brain-Tumor-Dataset/Training/{tumor_type}'
image_size = 128
batch_size = 32

def translate_classes(class_no:int):
     class_names = ['Glioma', 'Meningioma', 'No-Tumor', 'Pituitary\nTumor']
     return dict(zip(list(range(0,4)), class_names)).get(class_no)

def translate_datalabel(class_no:int):
     class_names = ['Glioma', 'Meningioma', 'Pituitary\nTumor', 'No-Tumor']
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
        df = llm.df
    return llm, pipeline, model, sample_image_path, selectable_images, df

llm, pipeline, model, sample_image_path, select_images, df = preprocess()


def validate_vlm(df):
    df = df.to_pandas().sample(n=10)
    st.write(df)
    bytes_arr = []
    labels_arr = []
    for idx, row in df.iterrows():
        img_path = row['image'].get('path')
        # archive = zipfile.ZipFile(img_path,'r')
        bytes_arr.append(img_path)
        labels_arr.append(translate_datalabel(row['label']))
    return bytes_arr, labels_arr
          


with st.sidebar:
    with st.form('main-form'):
        add_uni_logo = st.logo(COLUMBIA_LOGO, size='large')
        submitted = st.form_submit_button("Analyze Image")
        with st.expander("Deep Learning Prediction"):
            st.write("Sample Image for Analysis")
            sample_image = st.image(sample_image_path)
            if submitted:
                truth_label = st.markdown(f"Actual Tumor Type: {tumor_type}")
                img = image.load_img(sample_image_path)
                img_arr = image.img_to_array(img)
                resized = transform.resize(img_arr, (128,128,3))
                expanded_img = np.expand_dims(resized, axis=0)
                preprocessed_img = preprocess_input(expanded_img)
                prediction = model.predict(preprocessed_img)
                st.write(f"CNN Model Predicted Class: {translate_classes(np.argmax(prediction))}")
                st.write((prediction))
                st.divider()
                with st.spinner("Loading analysis..."):
                    llm_response = llm.analyze_image(llm.get_initial_analysis(img_path=sample_image_path))
                    st.write(f"Image Analysis: {llm_response}")
        
        st.divider()
        add_image = st.image(TRAINING_DATA_IMG, caption="Training Error", width=250)
        add_uploader = st.file_uploader(label="Upload image of interest to determine blastoma classification.", type=['png','jpg', 'pdf'])
        if add_uploader:
            name = add_uploader.name
            tmp_path = os.path.join('/tmp',name)
            os.makedirs("./tmp", exist_ok=True)
            with open(tmp_path, "wb") as file:
                 file.write(add_uploader.getvalue())
            st.toast(f"Temporary Location Developed: {tmp_path}")
            pipeline.add_docs(path=tmp_path)
            os.rmdir("./tmp")
        img_choices = image_select("Select image to analyze",select_images)
    st.divider()
    dataset_btn = st.button("Validate VLM Model")
    if dataset_btn:
        valid_df = pd.DataFrame(columns=['Prediction', 'Truth'])
        bytes_arr, label_arr = validate_vlm(df)
        for idx, byte in enumerate(bytes_arr[:5]):
            resp = llm.analyze_image(llm.dataset_analyze(byte))
            st.chat_message("ai").write(f"LLM Model Prediction: {resp} | Actual: {(label_arr[idx])}")
            temp_df = pd.DataFrame().from_dict({"Prediction": [resp], "Truth": [label_arr[idx]]})
            valid_df = pd.concat([valid_df, temp_df], ignore_index=True)
        valid_df.to_csv("vlm_model_res.csv")
        st.toast("Export complete!")

if "messages" not in st.session_state:
        st.session_state.messages = []
else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
# load_sidebar()

if prompt := st.chat_input("Enter your question"):
    with st.spinner("Synthesizing response.."):
        context = pipeline.retrieve_context(prompt)
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = llm.analyze_image(llm.extend_analysis(query=prompt, context=context, img_path=img_choices))
        st.chat_message("ai").write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.expander(label="Citations"):
             st.write(context)