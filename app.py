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

# --- GEMINI 2.5 FLASH API ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent"

def enhance_with_gemini(image, api_key):
    """Use Gemini 2.5 Flash to analyze and enhance the photo"""
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare the prompt for Gemini 2.5 Flash
        prompt = """
        You are a professional photo editor. Analyze this photo and provide enhancement parameters for a professional ID picture.
        
        Return ONLY this JSON format with exact numbers:
        {
            "brightness": 1.3,
            "contrast": 1.1,
            "shadows": 1.2,
            "highlights": 1.1,
            "skin_smooth": 5,
            "sharpness": 1.2,
            "redness_correction": -3,
            "saturation": 1.0,
            "clarity": 1.1
        }
        
        Guidelines:
        - brightness: 1.0 to 1.8 (higher = brighter) - gawing maliwanag ang mukha
        - contrast: 1.0 to 1.3
        - shadows: 1.0 to 1.5 (boost dark areas)
        - highlights: 0.9 to 1.2
        - skin_smooth: 3 to 8 (smoothing level)
        - sharpness: 1.0 to 1.5
        - redness_correction: -10 to 0 (negative = less red)
        - saturation: 0.9 to 1.2
        - clarity: 1.0 to 1.3
        
        Target: Professional ID photo na maliwanag ang mukha, natural ang skin tone, at parang studio quality.
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
        
        # Call Gemini 2.5 Flash API
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract text response
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
                    st.error("No JSON found in response")
                    return None
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Gemini 2.5 Flash Error: {str(e)}")
        return None

def apply_enhancements(image, params):
    """Apply the enhancements based on Gemini 2.5 Flash parameters"""
    
    if params is None:
        # Default enhanced parameters for bright professional look
        params = {
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
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Skin smoothing
    smooth_level = params.get('skin_smooth', 6)
    img = cv2.bilateralFilter(img, smooth_level, 50, 50)
    
    # 2. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 3. Advanced brightness and contrast
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Apply shadow boost and highlight control
    shadow_boost = params.get('shadows', 1.3)
    highlight_control = params.get('highlights', 1.1)
    
    # Equalize histogram first
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    
    # Apply brightness with shadow boost
    brightness = params.get('brightness', 1.5)
    yuv[:,:,0] = cv2.addWeighted(
        yuv[:,:,0], 
        brightness, 
        np.zeros_like(yuv[:,:,0]), 
        0, 
        15 * shadow_boost
    )
    
    # Clip values to valid range
    yuv[:,:,0] = np.clip(yuv[:,:,0], 0, 255).astype(np.uint8)
    
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Color correction (para hindi mamula)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply redness correction
    redness = params.get('redness_correction', -5)
    if redness != 0:
        a = cv2.addWeighted(a, 1.0, np.zeros_like(a), 0, redness)
    
    # Apply contrast with CLAHE
    contrast = params.get('contrast', 1.2)
    clahe = cv2.createCLAHE(clipLimit=contrast * 2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Apply clarity (local contrast enhancement)
    clarity = params.get('clarity', 1.2)
    if clarity > 1.0:
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]]) * (clarity - 1.0)
        l = cv2.filter2D(l, -1, kernel)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 5. Apply saturation
    saturation = params.get('saturation', 1.0)
    if saturation != 1.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * saturation, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 6. Convert back to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 7. Apply sharpness
    sharpness = params.get('sharpness', 1.3)
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(sharpness)
    
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
st.markdown("### Powered by Gemini 2.5 Flash - Ultra Bright Professional Quality")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key
    api_key = st.text_input("Gemini 2.5 Flash API Key", type="password", 
                           help="Get your free API key from Google AI Studio")
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key")
        st.markdown("[Get API Key](https://makersuite.google.com/app/apikey)")
    
    # Upload
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    # Options
    st.markdown("---")
    st.subheader("Options")
    
    col1, col2 = st.columns(2)
    with col1:
        bg_remove = st.checkbox("Remove Background", value=True)
    with col2:
        ultra_bright = st.checkbox("Ultra Bright Mode", value=True)
    
    size_option = st.selectbox(
        "Output Size",
        ["2x2 (600x600)", "1x1 (300x300)", "Passport (413x531)", "PRC Size (531x531)"]
    )
    
    enhance_btn = st.button("✨ Enhance with Gemini 2.5 Flash", type="primary", use_container_width=True)

# Main content
if uploaded_file and enhance_btn:
    if not api_key:
        st.error("❌ Please enter your Gemini API key")
    else:
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
                st.subheader("✨ Enhanced with Gemini 2.5 Flash")
                
                with st.spinner("🤖 Gemini 2.5 Flash is analyzing and enhancing your photo..."):
                    
                    # Get parameters from Gemini
                    params = enhance_with_gemini(original, api_key)
                    
                    if params:
                        st.success("✅ Gemini 2.5 Flash analysis complete!")
                        
                        # Override for ultra bright if checked
                        if ultra_bright:
                            params['brightness'] = min(params.get('brightness', 1.5) * 1.2, 1.9)
                            params['shadows'] = min(params.get('shadows', 1.3) * 1.1, 1.6)
                        
                        with st.expander("📊 AI Enhancement Parameters"):
                            st.json(params)
                    else:
                        st.warning("⚠️ Using default ultra bright settings")
                        params = None
                    
                    # Apply enhancements
                    enhanced = apply_enhancements(original, params)
                    
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
                        label="📥 Download Enhanced Photo",
                        data=buf.getvalue(),
                        file_name="gemini_enhanced_photo.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
                    
                    # Show improvement
                    improvement = ((brightness_final / brightness_orig) - 1) * 100
                    if improvement > 0:
                        st.success(f"✅ Brightness improved by {improvement:.0f}%!")
                    else:
                        st.success("✅ Photo enhanced successfully!")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # Welcome screen
    st.info("👆 Upload a photo and click 'Enhance with Gemini 2.5 Flash' to start")
    
    st.markdown("---")
    st.markdown("### ✨ Gemini 2.5 Flash Features:")
    st.markdown("""
    - ⚡ **2.5 Flash Technology** - Mas mabilis at mas matalino
    - 🔆 **Ultra Bright Mode** - Sobrang liwanag ng mukha
    - 🎨 **Smart Color Correction** - Hindi namumula
    - ✨ **Professional Studio Quality** - Parang kinuha sa studio
    - 🤖 **AI-Powered Analysis** - Perfect settings para sa photo mo
    
    **Get your free API key:** [Google AI Studio](https://makersuite.google.com/app/apikey)
    """)
