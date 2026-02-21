import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
from rembg import remove
import json
import base64

# Try to import Gemini, but don't fail if not installed
try:
    import google.generativeai as genai
    from google.generativeai.types import content_types
    from google.generativeai import protos
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("⚠️ Gemini AI not installed. Using standard enhancement only.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Pro AI ID Studio", page_icon="👤", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stImage {
        margin-bottom: 10px;
    }
    .element-container {
        margin-bottom: 10px !important;
    }
    .stButton button {
        width: 100%;
        margin-top: 10px;
    }
    .css-1r6slb0 {
        gap: 20px;
    }
    .css-183lzff {
        margin-bottom: 15px;
        text-align: center;
    }
    hr {
        margin: 20px 0 !important;
    }
    .gemini-badge {
        background: linear-gradient(135deg, #4285f4, #9b72cb);
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# --- GEMINI AI SETUP (with error handling) ---
@st.cache_resource
def setup_gemini(api_key):
    """Setup Gemini AI with API key"""
    if not GEMINI_AVAILABLE:
        return None
    
    try:
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        st.error(f"Gemini setup error: {str(e)}")
        return None

def analyze_image_with_gemini(image, model):
    """Use Gemini to analyze image and suggest improvements"""
    if not model or not GEMINI_AVAILABLE:
        return None
    
    try:
        # Create prompt for Gemini
        prompt = """
        Analyze this ID photo and provide specific adjustments needed:
        1. Brightness level (0-100) - current and recommended
        2. Contrast level (0-100)
        3. Skin tone issues (redness, paleness, etc.)
        4. Lighting problems (shadows, glare, etc.)
        5. Specific adjustments needed (in percentages)
        
        Format the response as JSON only:
        {
            "brightness": {"current": 0, "recommended": 0, "adjustment": 0},
            "contrast": {"current": 0, "recommended": 0, "adjustment": 0},
            "skin_tone_issues": ["issue1", "issue2"],
            "lighting_issues": ["issue1", "issue2"],
            "adjustments": {
                "brightness_boost": 0,
                "contrast_boost": 0,
                "color_correction": 0,
                "shadow_boost": 0
            }
        }
        """
        
        # Send to Gemini - pass PIL Image directly, not bytes
        response = model.generate_content([prompt, image])
        
        # Parse JSON response
        try:
            # Extract JSON from response
            response_text = response.text
            # Find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            return None
            
    except Exception as e:
        st.error(f"Gemini analysis error: {str(e)}")
        return None

def gemini_smart_enhance(image, adjustments):
    """Apply smart adjustments based on Gemini analysis"""
    if not adjustments:
        return ultra_bright_enhance(image, 1.8)
    
    # Get adjustments from Gemini
    brightness_boost = adjustments.get('adjustments', {}).get('brightness_boost', 20)
    contrast_boost = adjustments.get('adjustments', {}).get('contrast_boost', 10)
    
    # Convert to multiplier
    brightness_mult = 1.0 + (brightness_boost / 100)
    contrast_mult = 1.0 + (contrast_boost / 100)
    
    # Apply enhancements
    return ultra_bright_enhance(image, brightness_mult)

# --- SMART ENHANCEMENT FUNCTIONS ---
def smart_auto_enhance(image, brightness=1.2):
    """Automatic enhancement with adjustable brightness"""
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin Smoothing
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. Smart Brightness
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = cv2.addWeighted(yuv[:,:,0], brightness, np.zeros_like(yuv[:,:,0]), 0, 10)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Gamma correction for shadows
    if brightness > 1.2:
        gamma = 0.85
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
    
    # 5. Color correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 6. Convert to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 7. Final adjustments
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.1)
    
    # 8. Sharpening
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=0))
    
    return pil_img

def ultra_bright_enhance(image, brightness_multiplier=1.8):
    """Super bright enhancement with adjustable brightness"""
    
    # Convert to numpy array
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 1. Denoising
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    
    # 2. Skin Smoothing
    img = cv2.bilateralFilter(img, 5, 50, 50)
    
    # 3. Ultra Brightness
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    yuv[:,:,0] = cv2.addWeighted(yuv[:,:,0], brightness_multiplier, np.zeros_like(yuv[:,:,0]), 0, 15)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # 4. Gamma correction
    gamma = 0.8
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    
    # 5. Color correction
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 6. Convert to PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # 7. Aggressive brightness
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.3)
    
    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(1.15)
    
    # 8. Another brightness boost
    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(1.2)
    
    # 9. Sharpening
    pil_img = pil_img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=40, threshold=0))
    
    return pil_img

def resize_with_padding(image, target_size, fill_color=(255, 255, 255)):
    """Resize proportionally with white padding"""
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    new_image = Image.new("RGB", target_size, fill_color)
    left = (target_size[0] - image.size[0]) // 2
    top = (target_size[1] - image.size[1]) // 2
    
    new_image.paste(image, (left, top))
    return new_image

def process_id_photo(image, size_type, remove_bg):
    """Process ID photo with proper sizing and background"""
    
    if remove_bg:
        with st.spinner("Removing background..."):
            image_no_bg = remove(image)
            
            white_bg = Image.new("RGB", image_no_bg.size, (255, 255, 255))
            
            if image_no_bg.mode == 'RGBA':
                white_bg.paste(image_no_bg, mask=image_no_bg.split()[3])
            else:
                white_bg.paste(image_no_bg)
            
            image = white_bg
    
    # Standard Sizes
    sizes = {
        "2x2 (600x600 px)": (600, 600),
        "1x1 (300x300 px)": (300, 300),
        "Passport (413x531 px)": (413, 531),
        "PRC Size (531x531 px)": (531, 531)
    }
    
    target_size = sizes.get(size_type, (600, 600))
    
    if image.size[0] < target_size[0]:
        scale_factor = target_size[0] / image.size[0]
        new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return resize_with_padding(image, target_size)

# --- UI ---
st.title("👤 Pro AI ID Photo Studio")
st.markdown("### ✨ ULTRA BRIGHT Mode - Sobrang Liwanag ng Mukha")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Show Gemini status
    if not GEMINI_AVAILABLE:
        st.warning("⚠️ Install google-generativeai for AI features")
        st.code("pip install google-generativeai")
    
    # API Key input (only if Gemini available)
    api_key = ""
    if GEMINI_AVAILABLE:
        with st.expander("🔑 Gemini API Settings (Optional)", expanded=False):
            api_key = st.text_input("Enter Gemini API Key", type="password", 
                                   help="Get API key from https://makersuite.google.com/app/apikey")
            if api_key:
                st.success("✅ API Key set!")
    
    uploaded_file = st.file_uploader("📸 Upload Photo", type=["jpg", "png", "jpeg", "jfif"])
    
    size_opt = st.selectbox("📏 Select Output Size:", 
                           ["2x2 (600x600 px)", "1x1 (300x300 px)", 
                            "Passport (413x531 px)", "PRC Size (531x531 px)"])
    
    use_bg_rem = st.checkbox("🎯 Remove Background (White)", value=True)
    
    st.markdown("---")
    st.markdown("### 💡 Brightness Mode")
    
    # Enhancement options
    if GEMINI_AVAILABLE and api_key:
        enhance_mode = st.radio("Enhancement Mode:", 
                              ["🌐 Gemini AI Smart", "✨ Ultra Bright", "⚙️ Standard"],
                              index=1)  # Default to Ultra Bright
    else:
        enhance_mode = st.radio("Enhancement Mode:", 
                              ["✨ Ultra Bright", "⚙️ Standard"],
                              index=0)  # Default to Ultra Bright
    
    # Brightness control - defined for ALL modes except Gemini
    brightness_level = 1.8  # Default value
    
    if enhance_mode == "✨ Ultra Bright":
        brightness_level = st.slider("🔆 Ultra Bright Level", 1.0, 2.5, 1.8, 0.1,
                                    help="Higher = Mas maliwanag")
    elif enhance_mode == "⚙️ Standard":
        brightness_level = st.slider("🔆 Brightness Level", 1.0, 2.0, 1.2, 0.1,
                                    help="Adjust brightness level")
    
    generate_btn = st.button("✨ Generate Bright Photo", type="primary", use_container_width=True)

# Main Content
if uploaded_file:
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)
        
        if generate_btn:
            # Setup Gemini if available
            model = None
            if GEMINI_AVAILABLE and api_key and enhance_mode == "🌐 Gemini AI Smart":
                model = setup_gemini(api_key)
            
            # Analyze with Gemini if in AI mode
            adjustments = None
            if enhance_mode == "🌐 Gemini AI Smart" and model:
                with st.spinner("🤖 Gemini AI is analyzing your photo..."):
                    # Pass PIL Image directly, not bytes
                    adjustments = analyze_image_with_gemini(img, model)
                    if adjustments:
                        st.success("✅ AI Analysis complete!")
                        
                        # Show analysis results
                        with st.expander("📊 AI Analysis Results"):
                            col_a1, col_a2 = st.columns(2)
                            with col_a1:
                                brightness_adj = adjustments.get('adjustments', {}).get('brightness_boost', 0)
                                contrast_adj = adjustments.get('adjustments', {}).get('contrast_boost', 0)
                                st.metric("Brightness Boost", f"+{brightness_adj}%")
                                st.metric("Contrast Boost", f"+{contrast_adj}%")
                            with col_a2:
                                if adjustments.get('skin_tone_issues'):
                                    st.write("Skin issues:", ", ".join(adjustments['skin_tone_issues']))
                                if adjustments.get('lighting_issues'):
                                    st.write("Lighting issues:", ", ".join(adjustments['lighting_issues']))
                    else:
                        st.warning("⚠️ Gemini analysis failed. Using Ultra Bright mode instead.")
                        enhance_mode = "✨ Ultra Bright"  # Fallback to Ultra Bright
            
            # Create three columns
            col1, col2, col3 = st.columns(3, gap="large")
            
            with col1:
                st.markdown("### 📷 Original")
                st.image(img, use_container_width=True)
                st.caption(f"Size: {img.size[0]}x{img.size[1]} px")
                
                # Calculate original brightness
                img_np = np.array(img.convert('L'))
                avg_brightness = np.mean(img_np)
                st.caption(f"Original Brightness: {avg_brightness:.1f}/255")
                
                st.markdown("<br>", unsafe_allow_html=True)

            with col2:
                st.markdown("### ✨ Enhanced")
                with st.spinner("Applying enhancements..."):
                    
                    # Choose enhancement based on mode
                    if enhance_mode == "🌐 Gemini AI Smart" and adjustments:
                        enhanced_img = gemini_smart_enhance(img, adjustments)
                        caption = "🤖 Gemini AI Enhanced"
                    elif enhance_mode == "✨ Ultra Bright":
                        enhanced_img = ultra_bright_enhance(img, brightness_level)
                        caption = f"✨ Ultra Bright ({brightness_level}x)"
                    else:  # Standard mode
                        enhanced_img = smart_auto_enhance(img, brightness_level)
                        caption = f"⚙️ Standard Mode ({brightness_level}x)"
                    
                    st.image(enhanced_img, use_container_width=True)
                    st.caption(caption)
                    
                    # Show enhanced brightness
                    enhanced_np = np.array(enhanced_img.convert('L'))
                    enhanced_brightness = np.mean(enhanced_np)
                    st.caption(f"Enhanced Brightness: {enhanced_brightness:.1f}/255")
                    
                    st.markdown("<br>", unsafe_allow_html=True)

            with col3:
                st.markdown("### 🎯 Final ID")
                with st.spinner("Creating final photo..."):
                    final_result = process_id_photo(enhanced_img, size_opt, use_bg_rem)
                    st.image(final_result, use_container_width=True)
                    st.caption(f"{size_opt}")
                    
                    # Download button
                    buf = io.BytesIO()
                    final_result.save(buf, format="JPEG", quality=100, dpi=(300,300))
                    
                    st.download_button(
                        label="📥 Download Bright ID Photo",
                        data=buf.getvalue(),
                        file_name=f"bright_id_{size_opt.split()[0]}.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Success message
            brightness_increase = ((enhanced_brightness / avg_brightness) - 1) * 100
            st.success(f"✅ Success! Brightness increased by {brightness_increase:.0f}%")
            
            # Show what mode was used
            st.info(f"Used: {caption}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)  # Show full error for debugging
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Upload photo to begin")
        
        st.markdown("---")
        st.markdown("### ✨ Features:")
        st.markdown("""
        - 🔆 **ULTRA BRIGHT Mode** - Sobrang liwanag ng mukha (1.0x to 2.5x)
        - ⚙️ **Standard Mode** - Natural na liwanag (1.0x to 2.0x)
        - 🎨 **Natural skin tones** - Hindi namumula
        - 📸 **Professional finish** - Parang studio quality
        - 🤖 **Gemini AI ready** - Smart enhancement
        
        **Current Brightness:** 164.9/255 - Pwedeng pagandahin pa!
        """)
