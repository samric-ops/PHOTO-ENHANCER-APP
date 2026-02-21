import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageOps
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="📸", layout="wide")

# --- GEMINI API CONFIG ---
GEMINI_MODELS = {
    "Gemini 2.0 Flash (Fastest)": "gemini-2.0-flash-exp",
    "Gemini 1.5 Pro (Best Quality)": "gemini-1.5-pro",
    "Gemini 1.5 Flash (Balanced)": "gemini-1.5-flash"
}

def enhance_with_gemini_direct(image, api_key, model_name, instructions):
    """Let Gemini directly enhance the image based on instructions"""
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=100)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # API URL
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        # Prepare detailed prompt
        prompt = f"""
        You are a professional photo editor. Enhance this photo based on these EXACT instructions:
        
        {instructions}
        
        IMPORTANT: 
        - Return the ENHANCED IMAGE directly, not parameters
        - Make it look like a professional studio photo
        - The face should be VERY BRIGHT and clear
        - Skin tones should be natural (not red, not pale)
        - Remove any shadows on the face
        - Make it look like it was taken in a professional studio with perfect lighting
        
        Return the enhanced image.
        """
        
        # Prepare request
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 4096
            }
        }
        
        # Call API
        response = requests.post(
            f"{api_url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if image was returned
            if 'candidates' in result and len(result['candidates']) > 0:
                parts = result['candidates'][0]['content']['parts']
                for part in parts:
                    if 'inline_data' in part:
                        # Decode the image
                        img_data = base64.b64decode(part['inline_data']['data'])
                        enhanced_img = Image.open(io.BytesIO(img_data))
                        return enhanced_img
            
            return None
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def enhance_with_gemini_fallback(image, instructions):
    """Fallback enhancement if API fails - using PIL only"""
    
    # Convert to numpy for processing
    import cv2
    import numpy as np
    
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Aggressive enhancement
    # 1. Brightness boost
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = cv2.addWeighted(yuv[:,:,0], 1.8, np.zeros_like(yuv[:,:,0]), 0, 20)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 2. Gamma correction
    gamma = 0.7
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    
    # 3. Color correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Reduce redness
    a = cv2.addWeighted(a, 1.0, np.zeros_like(a), 0, -5)
    
    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 4. Sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Convert back
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# --- UI ---
st.title("📸 AI Photo Enhancer")
st.markdown("### ✨ Let Gemini AI Directly Enhance Your Photo")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    api_key = st.text_input("Gemini API Key", type="password", 
                           help="Get from https://makersuite.google.com/app/apikey")
    
    if not api_key:
        st.warning("⚠️ Enter API key to use AI enhancement")
    
    # Model selection
    selected_model = st.selectbox(
        "Gemini Model",
        list(GEMINI_MODELS.keys()),
        index=0
    )
    model_name = GEMINI_MODELS[selected_model]
    
    # Upload
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg"])
    
    st.markdown("---")
    st.subheader("🎨 Enhancement Style")
    
    # Enhancement instructions
    style = st.selectbox(
        "Select Style",
        [
            "Ultra Bright ID Photo",
            "Professional Studio Portrait",
            "Natural Bright",
            "Custom"
        ]
    )
    
    if style == "Custom":
        custom_instructions = st.text_area(
            "Custom Instructions",
            value="Make the face very bright and clear, remove shadows, natural skin tones, professional studio quality"
        )
    else:
        if style == "Ultra Bright ID Photo":
            instructions = "Make this an ULTRA BRIGHT ID photo. The face should be very bright and clear like it was taken in a professional studio. Remove all shadows. Skin should be natural (not red). Professional quality. The face should be the focus."
        elif style == "Professional Studio Portrait":
            instructions = "Enhance this to look like a professional studio portrait. Perfect lighting, bright face, clear details, natural skin tones. Remove any imperfections."
        elif style == "Natural Bright":
            instructions = "Make the face bright but keep it natural. Enhance the lighting, remove shadows, make it look professional but not over-processed."
    
    enhance_btn = st.button("✨ Enhance with AI", type="primary", use_container_width=True)

# Main content
if uploaded_file and enhance_btn:
    try:
        # Load image
        original = Image.open(uploaded_file)
        original = ImageOps.exif_transpose(original)
        
        # Create columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original")
            st.image(original, use_container_width=True)
            st.caption(f"Size: {original.size[0]}x{original.size[1]}")
        
        with col2:
            st.subheader("✨ AI Enhanced")
            
            if api_key and style != "Custom":
                with st.spinner(f"🤖 {selected_model} is enhancing your photo..."):
                    # Get instructions
                    if style == "Ultra Bright ID Photo":
                        inst = "Make this an ULTRA BRIGHT ID photo. The face should be very bright and clear like it was taken in a professional studio. Remove all shadows. Skin should be natural (not red). Professional quality. The face should be the focus."
                    elif style == "Professional Studio Portrait":
                        inst = "Enhance this to look like a professional studio portrait. Perfect lighting, bright face, clear details, natural skin tones. Remove any imperfections."
                    else:  # Natural Bright
                        inst = "Make the face bright but keep it natural. Enhance the lighting, remove shadows, make it look professional but not over-processed."
                    
                    # Try Gemini enhancement
                    enhanced = enhance_with_gemini_direct(original, api_key, model_name, inst)
                    
                    if enhanced:
                        st.image(enhanced, use_container_width=True)
                        st.success("✅ Enhanced by Gemini AI!")
                    else:
                        st.warning("⚠️ Using fallback enhancement")
                        enhanced = enhance_with_gemini_fallback(original, inst)
                        st.image(enhanced, use_container_width=True)
            else:
                # Fallback if no API key
                with st.spinner("Applying enhancement..."):
                    if style == "Custom":
                        inst = custom_instructions
                    else:
                        inst = instructions
                    
                    enhanced = enhance_with_gemini_fallback(original, inst)
                    st.image(enhanced, use_container_width=True)
                    if not api_key:
                        st.info("ℹ️ Using built-in enhancement (no API key)")
            
            # Download button
            buf = io.BytesIO()
            enhanced.save(buf, format="JPEG", quality=100)
            
            st.download_button(
                label="📥 Download Enhanced Photo",
                data=buf.getvalue(),
                file_name="ai_enhanced_photo.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    # Instructions
    st.info("👆 Upload a photo and click 'Enhance with AI'")
    
    st.markdown("---")
    st.markdown("### How to get the BEST results:")
    st.markdown("""
    1. **Get a free API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
    2. **Upload a clear photo** of your face
    3. **Select a style**:
       - **Ultra Bright ID Photo** - Sobrang liwanag, pang-ID
       - **Professional Studio Portrait** - Parang kinuha sa studio
       - **Natural Bright** - Maliwanag pero natural
    4. **Let Gemini do its magic** ✨
    
    The AI will directly enhance the image based on your instructions!
    """)
