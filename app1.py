import os
import tarfile
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# -------------------------------
# Constants
# -------------------------------
FROZEN_GRAPH_NAME = 'frozen_inference_graph.pb'
MODEL_TARBALL_PATH = 'deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz'


# -------------------------------
# DeepLab Model Loader
# -------------------------------
@st.cache_resource
def load_model(tarball_path):
    class DeepLabModel:
        INPUT_TENSOR_NAME = 'ImageTensor:0'
        OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

        def __init__(self, tarball_path):
            self.graph = tf.Graph()
            graph_def = None
            with tarfile.open(tarball_path) as tar:
                for tar_info in tar.getmembers():
                    if FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                        file_handle = tar.extractfile(tar_info)
                        graph_def = tf.GraphDef()
                        graph_def.ParseFromString(file_handle.read())
                        break
            if graph_def is None:
                raise RuntimeError('Cannot find inference graph in tar archive.')

            with self.graph.as_default():
                tf.import_graph_def(graph_def, name='')

            self.sess = tf.Session(graph=self.graph)

        def run(self, image):
            width, height = image.size
            resize_ratio = 513.0 / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            resized_image = image.convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
            seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={
                self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]
            })[0]
            return resized_image, seg_map

    return DeepLabModel(tarball_path)


# -------------------------------
# Wall Styling Function
# -------------------------------
def apply_wall_style(original_image, seg_map, color=None, texture_img=None, opacity=128):
    original_image = original_image.convert("RGBA")
    wall_mask = (seg_map == 1).astype(np.uint8) * 255
    wall_mask_image = Image.fromarray(wall_mask).resize(original_image.size)

    if texture_img:
        texture = texture_img.resize(original_image.size).convert("RGBA")
        texture.putalpha(opacity)
        styled_wall = Image.composite(texture, Image.new("RGBA", original_image.size), wall_mask_image)
    else:
        if color:
            styled_wall = Image.new("RGBA", original_image.size, color + (0,))
            styled_wall.putalpha(opacity)
            styled_wall = Image.composite(styled_wall, Image.new("RGBA", original_image.size), wall_mask_image)

    result = Image.alpha_composite(original_image, styled_wall)
    return result


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Wall Color Styler", layout="wide")
st.title("🎨 Wall Color Styler using DeepLabV3")

uploaded_image = st.file_uploader("Upload an interior image", type=['jpg', 'jpeg', 'png'])

use_texture = st.checkbox("Use texture instead of solid color")

# Opacity control
opacity = st.slider("Opacity (transparency)", min_value=0, max_value=255, value=130)

# -------------------------------
# Color selection
# -------------------------------
selected_color = None
preset_colors = {
    "Red": "#FF4B4B", "Blue": "#4B7BFF", "Green": "#4BFF84", "Orange": "#FF944B",
    "Teal": "#4BFFD1", "Purple": "#A84BFF", "Pink": "#FF4BE8", "Yellow": "#FFF24B",
    "Brown": "#8B4513", "Beige": "#F5F5DC", "Dark Gray": "#2F4F4F", "Light Gray": "#D3D3D3",
    "Sky Blue": "#87CEEB", "Navy Blue": "#001F3F", "Olive": "#808000", "Maroon": "#800000",
    "Charcoal": "#36454F", "Mint": "#98FF98", "Peach": "#FFDAB9", "Indigo": "#4B0082"
}

if not use_texture:
    st.markdown("**Choose a color:**")
    cols = st.columns(5)
    color_keys = list(preset_colors.keys())
    for i in range(0, 20, 5):
        for j in range(5):
            name = color_keys[i + j]
            with cols[j]:
                if st.button("", key=name, help=name):
                    selected_color = tuple(int(preset_colors[name].lstrip("#")[k:k+2], 16) for k in (0, 2, 4))
                st.markdown(f'<div style="width:30px;height:30px;border-radius:50%;background-color:{preset_colors[name]};margin:auto;"></div>', unsafe_allow_html=True)

    st.markdown("---")
    custom = st.color_picker("Or pick a custom color", "#0080ff")
    if st.button("Use Custom Color"):
        selected_color = tuple(int(custom.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

# -------------------------------
# Texture upload
# -------------------------------
texture_img = None
if use_texture:
    texture_file = st.file_uploader("Upload texture image", type=['jpg', 'jpeg', 'png'])
    if texture_file:
        texture_img = Image.open(texture_file)

# -------------------------------
# Main processing
# -------------------------------
if uploaded_image and (selected_color or texture_img):
    with st.spinner("Processing image..."):
        model = load_model(MODEL_TARBALL_PATH)
        input_img = Image.open(uploaded_image)
        resized_img, seg_map = model.run(input_img)

        output_img = apply_wall_style(
            resized_img,
            seg_map,
            color=selected_color,
            texture_img=texture_img,
            opacity=opacity
        )

    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.image(resized_img, caption="Original Resized", use_column_width=True)
    with col2:
        st.image(output_img, caption="Styled Output", use_column_width=True)
