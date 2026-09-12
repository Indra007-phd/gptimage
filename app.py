import streamlit as st
from openai import OpenAI
import base64
import io
import zipfile
import requests
from PIL import Image

# Set page config
st.set_page_config(page_title="AI Image Modifier", page_icon="🎨", layout="wide")

# Initialize session state for storing results
if "processed_images" not in st.session_state:
    st.session_state.processed_images = []

def encode_image(uploaded_file):
    """Encode the uploaded image to base64 for GPT-4o Vision."""
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

def generate_modified_image(client, image_file, user_prompt):
    """Pipeline: GPT-4o analyzes image + prompt -> DALL-E 3 generates new image."""
    base64_image = encode_image(image_file)
    
    # Step 1: Use GPT-4o to analyze the image and incorporate the user's prompt
    vision_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"You are an expert prompt engineer. Analyze this image in detail. Then, apply the following user request to modify it: '{user_prompt}'. Output ONLY a highly detailed DALL-E 3 image generation prompt that recreates this image with the requested modifications. Do not include any conversational text."
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]
    )
    
    dalle_prompt = vision_response.choices[0].message.content.strip()
    
    # Step 2: Generate the new image using DALL-E 3
    image_response = client.images.generate(
        model="dall-e-3",
        prompt=dalle_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )
    
    image_url = image_response.data[0].url
    
    # Fetch the actual image data
    img_data = requests.get(image_url).content
    return img_data

def create_zip(image_data_list):
    """Package multiple images into a single ZIP file."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img_data in enumerate(image_data_list):
            zip_file.writestr(f"modified_image_{i+1}.png", img_data)
    return zip_buffer.getvalue()

# --- UI Layout ---
st.title("🎨 AI Image Modifier")
st.markdown("Upload one or multiple images, provide a prompt, and let ChatGPT modify them.")

# Sidebar for API Key
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key here.")
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("1. Upload image(s)\n2. Enter a modification prompt\n3. GPT-4o interprets the image + prompt\n4. DALL-E 3 generates the result")

# Main content
uploaded_files = st.file_uploader("Upload Image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
user_prompt = st.text_input("Enter your modification prompt:", placeholder="e.g., Change the background to a futuristic city, make it cyberpunk style...")

if st.button("Modify Image(s)", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one image.")
    elif not user_prompt:
        st.warning("Please enter a prompt to modify the image.")
    else:
        client = OpenAI(api_key=api_key)
        st.session_state.processed_images = [] # Clear previous results
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing image {i+1} of {len(uploaded_files)}...")
            try:
                # Generate modified image
                modified_img_data = generate_modified_image(client, file, user_prompt)
                st.session_state.processed_images.append({
                    "original_name": file.name,
                    "original_data": file.getvalue(),
                    "modified_data": modified_img_data
                })
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            
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