import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image
import io

# Check Torch and CUDA availability
st.sidebar.title("System Info")
st.sidebar.write(f"Torch Version: {torch.__version__}")
st.sidebar.write(f"CUDA Available: {torch.cuda.is_available()}")

# Load translation model
@st.cache_resource
def load_translation_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    return pipeline('translation', model=model, tokenizer=tokenizer, src_lang="tam_Taml", tgt_lang="eng_Latn", max_length=400)

translator = load_translation_model()

# Load Stable Diffusion model
@st.cache_resource
def load_stable_diffusion():
    model_name = "CompVis/stable-diffusion-v1-4"
    pipeline_model = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pipeline_model.to(device)

sd_pipeline = load_stable_diffusion()

# App title
st.title("Tamil to English Translator & Image Generator")

# Input field for Tamil text
tamil_text = st.text_input("Enter Tamil Text", "")

import time

if st.button("Translate & Generate Image"):
    if tamil_text:
        # Measure translation time
        start_time = time.time()
        with st.spinner("Translating..."):
            translation_result = translator(tamil_text)
            english_text = translation_result[0]["translation_text"]
        translation_time = time.time() - start_time  # Time taken for translation
        
        st.subheader("Translated English Text")
        st.write(english_text)
        st.write(f"Translation Time: {translation_time:.2f} seconds")

        # Measure image generation time
        start_time = time.time()
        with st.spinner("Generating Image..."):
            image = sd_pipeline(english_text, num_inference_steps=50).images[0]
        image_generation_time = time.time() - start_time  # Time taken for image generation
        
        # Display the image and time
        st.subheader("Generated Image")
        st.image(image, caption="Generated Image", use_column_width=True)
        st.write(f"Image Generation Time: {image_generation_time:.2f} seconds")
    else:
        st.warning("Please enter Tamil text.")


# Footer
st.markdown("---")
st.markdown("Developed using Streamlit, Transformers, and Stable Diffusion")


# Text to use: சூரிய அஸ்தமனத்துடன் மலை மற்றும் நதியுடன் கூடிய கற்பனை நிலப்பரப்பு