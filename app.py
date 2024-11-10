import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom right, #ff7e5f, #feb47b);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > input {
        border: 2px solid #ff7e5f;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #333;
        background-color: #fff;
    }
    .stButton button {
        background-color: #ff7e5f;
        color: white;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #feb47b;
        color: #333;
    }
    .header {
        text-align: center;
        padding: 20px;
        color: #fff;
    }
    .footer {
        text-align: center;
        color: #fff;
        font-size: 14px;
        padding-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the Stable Diffusion Model
@st.cache_resource
def load_pipeline(model_id):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, use_safetensors=True
    )
    # Check if CUDA is available, otherwise use CPU
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    return pipe

# Initialize the model
MODEL_ID = "dreamlike-art/dreamlike-diffusion-1.0"
pipe = load_pipeline(MODEL_ID)

# App Title
st.markdown("<h1 class='header'>🎨 AI Art Generator</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Create stunning images from your text prompts using Stable Diffusion</h3>", unsafe_allow_html=True)

# Input Section
prompt = st.text_input("📝 Enter your text prompt:", value="A futuristic city with flying cars at sunset")

# Advanced Options with Custom Styling
with st.expander("⚙️ Advanced Settings", expanded=True):
    st.markdown("<p style='color: #fff;'>Customize the image size below:</p>", unsafe_allow_html=True)
    width = st.slider("Image Width", 256, 1024, 512)
    height = st.slider("Image Height", 256, 1024, 512)

# Generate Image Button
if st.button("✨ Generate Image"):
    with st.spinner("Creating your masterpiece... 🎨"):
        try:
            params = {
                "width": width,
                "height": height,
            }
            result = pipe(prompt, **params)
            image = result.images[0]

            # Display Image
            st.image(image, caption="Generated Image", use_column_width=True)

            # Download Option
            image.save("generated_image.png")
            st.download_button("📥 Download Image", data=open("generated_image.png", "rb"), file_name="generated_image.png", mime="image/png")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Footer
st.markdown("<div class='footer'>Powered by Stable Diffusion and Streamlit | Made with ❤️ by Danish Shahzad</div>", unsafe_allow_html=True)
