import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove
import base64
import requests
import json

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Photo Enhancer", page_icon="📸", layout="wide")

# --- GEMINI AVAILABLE MODELS ---
# List of available models: https://ai.google.dev/gemini-api/docs/models
AVAILABLE_MODELS = {
    "Gemini 1.5 Pro": "gemini-1.5-pro",
    "Gemini 1.5 Flash": "gemini-1.5-flash",
    "Gemini 1.5 Flash-8B": "gemini-1.5-flash-8b",
    "Gemini 2.0 Flash": "gemini-2.0-flash-exp",  # Latest available
}

def enhance_with_gemini(image, api_key, model_name):
    """Use Gemini to analyze and enhance the photo"""
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # API URL with selected model
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        # Prepare the prompt
        prompt = """
        You are a professional photo editor. Analyze this photo and provide enhancement parameters for a professional ID picture.
        
        Return ONLY this JSON format with exact numbers:
        {
            "brightness": 1.5,
            "contrast": 1.2,
            "shadows": 1.3,
            "highlights": 1.1,
            "skin_smooth": 6,
            "sharpness": 1.3,
            "redness_correction": -5,
            "saturation": 1.0,
            "clarity": 1.2
        }
        
        Guidelines:
        - brightness: 1.0 to 1.8 (higher = brighter) - gawing VERY MALIWANAG ang mukha
        - contrast: 1.0 to 1.3
        - shadows: 1.0 to 1.5 (boost dark areas)
        - skin_smooth: 3 to 8 (smoothing level)
        - sharpness: 1.0 to 1.5
        - redness_correction: -10 to 0 (negative = less red)
        
        Target: ULTRA BRIGHT professional ID photo na parang studio quality.
        """
        
        # Prepare the request
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
                "temperature": 0.2,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 2048
            }
        }
        
        # Call Gemini API
        response = requests.post(
            f"{api_url}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                text_response = result['candidates'][0]['content']['parts'][0]['text']
                
                # Find JSON in response
                start_idx = text_response.find('{')
                end_idx = text_response.rfind('}') + 1
                
                if start_idx != -1 and end_idx > start_idx:
                    json_str = text_response[start_idx:end_idx]
                    params = json.loads(json_str)
                    return params
        else:
            st.warning(f"API Error: Using default settings")
            return None
            
    except Exception as e:
        st.warning(f"Gemini API Error: Using default settings")
        return None

def ultra_bright_enhance(image, params=None):
    """Ultra bright enhancement para sobrang liwanag ng mukha"""
    
    if params is None:
        # Default ultra bright parameters
        params = {
            "brightness": 1.7,
            "contrast": 1.2,
            "shadows": 1.4,
            "highlights": 1.1,
            "skin_smooth": 6,
            "sharpness": 1.3,
            "redness_correction": -5,
            "saturation": 1.0,
            "clarity": 1.2
        }
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Skin smoothing
    img = cv2.bilateralFilter(img, params.get('skin_smooth', 6), 50, 50)
    
    # 2. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 3. ULTRA BRIGHTNESS - multiple layers
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # First pass: histogram equalization
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    
    # Second pass: aggressive brightness boost
    brightness = params.get('brightness', 1.7)
    shadow_boost = params.get('shadows', 1.4)
    yuv[:,:,0] = cv2.addWeighted(
        yuv[:,:,0], 
        brightness, 
        np.zeros_like(yuv[:,:,0]), 
        0, 
        20 * shadow_boost
    )
    
    # Clip to valid range
    yuv[:,:,0] = np.clip(yuv[:,:,0], 0, 255).astype(np.uint8)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Gamma correction (para lumiwanag ang shadows)
    gamma = 0.75  # <1 = brighter
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    
    # 5. Color correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Redness correction
    redness = params.get('redness_correction', -5)
    if redness != 0:
        a = cv2.addWeighted(a, 1.0, np.zeros_like(a), 0, redness)
    
    # Contrast enhancement
    contrast = params.get('contrast', 1.2)
    clahe = cv2.createCLAHE(clipLimit=contrast * 2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 6. Convert to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 7. Final brightness boost
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    # 8. Sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(params.get('sharpness', 1.3))
    
    return pil_img

def remove_background(image):
    """Remove background and replace with white"""
    try:
        image_no_bg = remove(image)
        white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
        if image_no_bg.mode == 'RGBA':
            white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
        else:
            white_bg.paste(image_no_bg)
        return white_bg
    except:
        return image

def resize_to_id(image, size_type):
    """Resize to standard ID sizes"""
    sizes = {
        "2x2 (600x600)": (600, 600),
        "1x1 (300x300)": (300, 300),
        "Passport (413x531)": (413, 531),
        "PRC Size (531x531)": (531, 531)
    }
    target = sizes.get(size_type, (600, 600))
    
    # Resize proportionally
    image.thumbnail(target, Image.Resampling.LANCZOS)
    
    # Add white padding
    new_img = Image.new("RGB", target, (255, 255, 255))
    left = (target[0] - image.size[0]) // 2
    top = (target[1] - image.size[1]) // 2
    new_img.paste(image, (left, top))
    
    return new_img

# --- UI ---
st.title("📸 AI Photo Enhancer")
st.markdown("### ⚡ ULTRA BRIGHT Mode - Sobrang Liwanag ng Mukha")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    api_key = st.text_input("Gemini API Key", type="password", 
                           help="Get your free API key from Google AI Studio")
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key")
        st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")
    
    # Model selection
    selected_model = st.selectbox(
        "Gemini Model",
        list(AVAILABLE_MODELS.keys()),
        index=1  # Default to 1.5 Flash
    )
    model_name = AVAILABLE_MODELS[selected_model]
    
    # Upload
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    # Options
    st.markdown("---")
    st.subheader("Options")
    
    bg_remove = st.checkbox("Remove Background (White)", value=True)
    
    # Brightness level slider
    brightness_level = st.slider(
        "Ultra Bright Level", 
        1.0, 2.2, 1.8, 0.1,
        help="Mas mataas = Mas maliwanag ang mukha"
    )
    
    size_option = st.selectbox(
        "Output Size",
        ["2x2 (600x600)", "1x1 (300x300)", "Passport (413x531)", "PRC Size (531x531)"]
    )
    
    enhance_btn = st.button("✨ Enhance Photo", type="primary", use_container_width=True)

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
            
            # Show info
            st.caption(f"Size: {original.size[0]}x{original.size[1]}")
            
            # Calculate brightness
            img_gray = np.array(original.convert('L'))
            brightness_orig = np.mean(img_gray)
            st.caption(f"Brightness: {brightness_orig:.1f}/255")
        
        with col2:
            st.subheader("✨ Ultra Bright Enhanced")
            
            with st.spinner(f"🤖 Using {selected_model} for analysis..."):
                
                # Try to get AI parameters if API key is provided
                params = None
                if api_key:
                    params = enhance_with_gemini(original, api_key, model_name)
                    if params:
                        st.success("✅ AI analysis complete!")
                        # Override brightness with slider
                        params['brightness'] = brightness_level
                    else:
                        st.info("⚡ Using Ultra Bright default settings")
                        params = {"brightness": brightness_level}
                else:
                    params = {"brightness": brightness_level}
                
                # Apply ultra bright enhancement
                enhanced = ultra_bright_enhance(original, params)
                
                # Remove background if requested
                if bg_remove:
                    enhanced = remove_background(enhanced)
                
                # Resize
                final = resize_to_id(enhanced, size_option)
                
                st.image(final, use_container_width=True)
                
                # Show enhanced brightness
                final_gray = np.array(final.convert('L'))
                brightness_final = np.mean(final_gray)
                st.caption(f"Enhanced Brightness: {brightness_final:.1f}/255")
                
                # Download button
                buf = io.BytesIO()
                final.save(buf, format="JPEG", quality=100)
                
                st.download_button(
                    label="📥 Download Ultra Bright Photo",
                    data=buf.getvalue(),
                    file_name="ultra_bright_photo.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
                # Show improvement
                improvement = ((brightness_final / brightness_orig) - 1) * 100
                if improvement > 0:
                    st.success(f"✅ Brightness improved by {improvement:.0f}%!")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    # Welcome screen
    st.info("👆 Upload a photo and click 'Enhance Photo' to start")
    
    st.markdown("---")
    st.markdown("### ⚡ Ultra Bright Features:")
    st.markdown("""
    - 🔆 **ULTRA BRIGHT** - Sobrang liwanag ng mukha (adjustable 1.0x to 2.2x)
    - 🤖 **AI-Powered** - Using Gemini 1.5 Flash/Pro
    - 🎨 **Natural Skin Tones** - Hindi namumula
    - ✨ **Studio Quality** - Professional finish
    - 🖼️ **ID Sizes** - 2x2, 1x1, Passport, PRC
    
    **Available Models:**
    - Gemini 1.5 Flash (Mabilis)
    - Gemini 1.5 Pro (Matalino)
    - Gemini 2.0 Flash (Experimental)
    """)
