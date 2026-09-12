import streamlit as st
from gradio_client import Client, handle_file
import zipfile
import io
import os
import tempfile
from PIL import Image

# Set page config
st.set_page_config(page_title="Free AI Image Modifier", page_icon="🎨", layout="wide")

# Initialize session state for storing results
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []

def generate_modified_image(image_bytes, user_prompt):
    """Uses a free Hugging Face server (Instruct-Pix2Pix) - No API Key needed!"""
    
    # The Gradio client requires a physical file path, so we save the upload temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_in:
        temp_in.write(image_bytes)
        temp_in_path = temp_in.name

    try:
        # Connect to the public, completely free Instruct-Pix2Pix Hugging Face space
        client = Client("timbrooks/instruct-pix2pix")
        
        # Send the image and prompt to the free server
        result = client.predict(
            prompt=user_prompt,
            image=handle_file(temp_in_path),
            text_cfg_scale=7.5,
            image_cfg_scale=1.5,
            randomize_seed=True,
            seed=0,
            api_name="/generate"
        )
        
        # Gradio returns the file path to the newly generated image
        output_path = result[0] if isinstance(result, tuple) else result
        
        with open(output_path, "rb") as f:
            output_bytes = f.read()
            
        return output_bytes
        
    finally:
        # Clean up the temporary input file from the server
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)

def create_zip(image_data_list):
    """Package multiple images into a single ZIP file."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_data in enumerate(image_data_list):
            zip_file.writestr(f"modified_image_{i+1}.png", img_data)
    return zip_buffer.getvalue()

# --- UI Layout ---
st.title("🎨 Free AI Image Modifier")
st.markdown("Upload images, type an instruction (e.g., 'Make it look like a watercolor painting'), and modify them instantly for free.")

# Main content
uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
user_prompt = st.text_input("Enter your modification prompt:", placeholder="e.g., Change the background to a futuristic city, make it cyberpunk style...")

if st.button("Modify Image(s)", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one image.")
    elif not user_prompt:
        st.warning("Please enter a prompt to modify the image.")
    else:
        st.session_state.processed_images = [] # Clear previous results
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1} of {len(uploaded_files)}... (This may take 10-20 seconds on the free server)")
            try:
                # Generate modified image
                modified_img_data = generate_modified_image(file.getvalue(), user_prompt)
                st.session_state.processed_images.append({
                    "original_name": file.name,
                    "original_data": file.getvalue(),
                    "modified_data": modified_img_data
                })
            except Exception as e:
                st.error(f"Error processing {file.name}. The free server might be busy, please try again. Details: {str(e)}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Processing complete!")

# --- Display Results and Download ---
if st.session_state.processed_images:
    st.markdown("### Results")
    
    # Display images side-by-side
    for item in st.session_state.processed_images:
        col1, col2 = st.columns(2)
        with col1:
            st.image(item["original_data"], caption=f"Original: {item['original_name']}", use_column_width=True)
        with col2:
            st.image(item["modified_data"], caption=f"Modified: {item['original_name']}", use_column_width=True)
            
    st.markdown("---")
    
    # Download options
    if len(st.session_state.processed_images) == 1:
        # Single download
        img_data = st.session_state.processed_images[0]["modified_data"]
        st.download_button(
            label="Download Modified Image",
            data=img_data,
            file_name=f"modified_{st.session_state.processed_images[0]['original_name']}",
            mime="image/png"
        )
    else:
        # Bulk download (ZIP)
        img_data_list = [item["modified_data"] for item in st.session_state.processed_images]
        zip_data = create_zip(img_data_list)
        st.download_button(
            label="Download All Images as ZIP",
            data=zip_data,
            file_name="modified_images.zip",
            mime="application/zip"
        )
