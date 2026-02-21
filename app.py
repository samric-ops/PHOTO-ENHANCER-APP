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

# --- GEMINI 2.5 API CONFIGURATION ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent"

def enhance_with_gemini_25(image, api_key):
    """Use Gemini 2.5 to analyze and enhance the photo"""
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare the prompt for Gemini 2.5
        prompt = """
        Analyze this photo and provide enhancement parameters for a professional ID photo.
        The photo should look like it was taken in a professional studio - crisp, clear, and well-lit.
        
        Return ONLY a JSON object with these exact parameters:
        {
            "brightness_boost": 1.3,
            "contrast_boost": 1.1,
            "color_enhancement": 1.0,
            "sharpness": 1.2,
            "skin_smoothing": 5,
            "redness_correction": -5,
            "shadow_boost": 1.2
        }
        
        Base your recommendations on the current photo quality.
        """
        
        # Prepare the request
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
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
            }]
        }
        
        # Call Gemini 2.5 API
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            # Extract the text response
            text_response = result['candidates'][0]['content']['parts'][0]['text']
            
            # Find JSON in the response
            start_idx = text_response.find('{')
            end_idx = text_response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                params = json.loads(text_response[start_idx:end_idx])
                return params
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"Gemini 2.5 Error: {str(e)}")
        return None

def apply_enhancements(image, params):
    """Apply the enhancements based on Gemini 2.5 parameters"""
    
    if params is None:
        # Default parameters if Gemini fails
        params = {
            "brightness_boost": 1.3,
            "contrast_boost": 1.1,
            "color_enhancement": 1.0,
            "sharpness": 1.2,
            "skin_smoothing": 5,
            "redness_correction": -5,
            "shadow_boost": 1.2
        }
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Skin smoothing (using bilateral filter)
    img = cv2.bilateralFilter(img, params.get('skin_smoothing', 5), 50, 50)
    
    # 2. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 3. Brightness and contrast enhancement
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # Apply shadow boost
    shadow_boost = params.get('shadow_boost', 1.2)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = cv2.addWeighted(
        yuv[:,:,0], 
        params.get('brightness_boost', 1.3), 
        np.zeros_like(yuv[:,:,0]), 
        0, 
        10 * shadow_boost
    )
    
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Color correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply redness correction
    redness_correction = params.get('redness_correction', -5)
    if redness_correction != 0:
        a = cv2.addWeighted(a, 1.0, np.zeros_like(a), 0, redness_correction)
    
    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=params.get('contrast_boost', 1.1) * 1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 5. Convert back to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 6. Apply color enhancement if needed
    if params.get('color_enhancement', 1.0) != 1.0:
        enhancer = ImageEnhance.Color(pil_img)
        pil_img = enhancer.enhance(params['color_enhancement'])
    
    # 7. Apply sharpness
    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(params.get('sharpness', 1.2))
    
    return pil_img

def remove_background(image):
    """Remove background and replace with white"""
    image_no_bg = remove(image)
    white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
    if image_no_bg.mode == 'RGBA':
        white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
    else:
        white_bg.paste(image_no_bg)
    return white_bg

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
st.markdown("### Powered by Gemini 2.5 - Professional Studio Quality")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # API Key (required for Gemini 2.5)
    api_key = st.text_input("Gemini 2.5 API Key", type="password", 
                           help="Get your API key from Google AI Studio")
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini 2.5 API key")
    
    # Upload
    uploaded_file = st.file_uploader("Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    # Options
    st.markdown("---")
    st.subheader("Options")
    
    bg_remove = st.checkbox("Remove Background (White)", value=True)
    
    size_option = st.selectbox(
        "Output Size",
        ["2x2 (600x600)", "1x1 (300x300)", "Passport (413x531)", "PRC Size (531x531)"]
    )
    
    enhance_btn = st.button("✨ Enhance with Gemini 2.5", type="primary", use_container_width=True)

# Main content
if uploaded_file and enhance_btn:
    if not api_key:
        st.error("❌ Please enter your Gemini 2.5 API key")
    else:
        try:
            # Load image
            original = Image.open(uploaded_file)
            original = ImageOps.exif_transpose(original)
            
            # Create columns for before/after
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original")
                st.image(original, use_container_width=True)
                
                # Show info
                st.caption(f"Size: {original.size[0]}x{original.size[1]}")
                
                # Calculate brightness
                img_gray = np.array(original.convert('L'))
                brightness = np.mean(img_gray)
                st.caption(f"Brightness: {brightness:.1f}/255")
            
            with col2:
                st.subheader("✨ Enhanced with Gemini 2.5")
                
                with st.spinner("🤖 Gemini 2.5 is analyzing and enhancing your photo..."):
                    # Step 1: Get enhancement parameters from Gemini 2.5
                    params = enhance_with_gemini_25(original, api_key)
                    
                    if params:
                        st.success("✅ Gemini 2.5 analysis complete!")
                        
                        # Show parameters used
                        with st.expander("📊 Enhancement Parameters"):
                            st.json(params)
                        
                        # Step 2: Apply enhancements
                        enhanced = apply_enhancements(original, params)
                        
                        # Step 3: Remove background if requested
                        if bg_remove:
                            enhanced = remove_background(enhanced)
                        
                        # Step 4: Resize
                        final = resize_to_id(enhanced, size_option)
                        
                        st.image(final, use_container_width=True)
                        
                        # Show enhanced brightness
                        final_gray = np.array(final.convert('L'))
                        final_brightness = np.mean(final_gray)
                        st.caption(f"Enhanced Brightness: {final_brightness:.1f}/255")
                        
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
                        improvement = ((final_brightness / brightness) - 1) * 100
                        st.success(f"✅ Brightness improved by {improvement:.0f}%")
                    else:
                        st.error("❌ Gemini 2.5 analysis failed. Please try again.")
                        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

else:
    # Welcome screen
    st.info("👆 Upload a photo and click 'Enhance with Gemini 2.5' to start")
    
    st.markdown("---")
    st.markdown("### ✨ What Gemini 2.5 will do:")
    st.markdown("""
    - 🤖 **Smart Analysis** - Gemini 2.5 analyzes your photo
    - 🎯 **Custom Parameters** - AI determines the best settings
    - 🔆 **Perfect Brightness** - Maliwanag pero natural
    - 🎨 **Color Correction** - Hindi namumula ang mukha
    - ✨ **Professional Quality** - Parang studio photo
    
    **Get your free API key:** [Google AI Studio](https://makersuite.google.com/app/apikey)
    """)
