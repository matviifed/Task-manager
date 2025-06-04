import os
import sys
import cv2
import numpy as np
import json
import argparse
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from collections import Counter
from torchvision import models
import os
import re
from torchvision.transforms.functional import pad
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

sys.stdout.reconfigure(encoding='utf-8')
sys.stdin.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
from transformers import pipeline
from brand_training import LogoClassificationModel

try:
    import webcolors
    HAS_WEBCOLORS = True
except ImportError:
    HAS_WEBCOLORS = False
    print("webcolors module not available, using fallback color system", file=sys.stderr)

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("transformers not available, using templated descriptions", file=sys.stderr)

CLOTHING_CATEGORIES = [
    'top', 't-shirt', 'dress', 'pants', 'jeans', 'shorts', 'skirt', 'coat', 
    'jacket', 'sweater', 'hoodie', 'blouse', 'shirt', 'suit', 'shoes', 
    'sneakers', 'boots', 'hat', 'cap', 'scarf', 'tie', 'handbag',
    'backpack', 'sunglasses', 'watch', 'jewelry', 'belt', 'socks'
]

STYLES = [
    'casual', 'formal', 'streetwear', 'business', 'athletic', 'vintage', 
    'bohemian', 'minimalist', 'preppy', 'punk', 'hipster', 'chic',
    'professional', 'elegant', 'retro', 'sporty', 'urban', 'classic'
]

PATTERNS = [
    'solid', 'striped', 'checkered', 'plaid', 'floral', 'polka dot', 
    'geometric', 'abstract', 'print', 'graphic', 'tie-dye', 'gradient'
]

MATERIALS = [
    'cotton', 'denim', 'leather', 'silk', 'wool', 'polyester', 'linen', 
    'cashmere', 'jersey', 'suede', 'velvet', 'tweed', 'fleece', 'satin', 
    'chiffon', 'knit', 'nylon', 'chambray'
]

SEASONS = ['spring', 'summer', 'fall', 'winter', 'all-season']

OCCASIONS = [
    'everyday', 'work', 'party', 'formal event', 'casual outing', 
    'workout', 'date night', 'beach', 'outdoor', 'travel', 'lounge'
]

FALLBACK_COLORS = {
    (0, 0, 0): "black",
    (255, 255, 255): "white",
    (255, 0, 0): "red",
    (0, 255, 0): "green",
    (0, 0, 255): "blue",
    (255, 255, 0): "yellow",
    (255, 0, 255): "magenta",
    (0, 255, 255): "cyan",
    (128, 128, 128): "gray",
    (128, 0, 0): "maroon",
    (0, 128, 0): "darkgreen",
    (0, 0, 128): "navy",
    (128, 128, 0): "olive",
    (128, 0, 128): "purple",
    (0, 128, 128): "teal",
    (165, 42, 42): "brown",
    (250, 235, 215): "beige",
    (255, 192, 203): "pink",
    (255, 165, 0): "orange"
}

FASHION_BRANDS = ['Adidas', 'Puma', 'NewBalance', 'prada', 'lacoste', 'Gucci', 'louis vuitton', 
                  'Balenciaga', 'Chrome Hearts', 'Converse', 'Columbia', 'Hugo Boss', 'levis', 
                  'Ralph Lauren', 'versace', 'zara']

class ClothingImageAnalyzer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}", file=sys.stderr)
        
        self.setup_models()
        
        self.setup_logo_detection_model()
        
        self.text_generator = pipeline(
            "text-generation",
            model="gpt2-medium",
            tokenizer="gpt2-medium",
            max_length=180,
            top_k=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256,
            truncation=True,
            device=0 if torch.cuda.is_available() else -1
        )

    def setup_models(self):
        print("Loading ResNet model...", file=sys.stderr)
        
        weights = ResNet18_Weights.DEFAULT
        self.feature_model = resnet18(weights=weights)
        self.feature_model = torch.nn.Sequential(*list(self.feature_model.children())[:-1])
        self.feature_model.to(self.device)
        self.feature_model.eval()
        
        self.preprocess = weights.transforms()
        
        print("Loading pre-trained Fashion MNIST model...", file=sys.stderr)
        self.fashion_model = models.resnet18(weights=None)  
        self.fashion_model.fc = nn.Linear(self.fashion_model.fc.in_features, 10) 
        
        try:
            self.fashion_model.load_state_dict(torch.load("fashion_mnist_resnet50.pth", 
                                                        map_location=self.device))
            print("Successfully loaded pre-trained fashion model", file=sys.stderr)
        except Exception as e:
            print(f"Error loading fashion model: {e}", file=sys.stderr)
            print("Using simulated fashion classifier instead", file=sys.stderr)
        
        self.fashion_model.to(self.device)
        self.fashion_model.eval()
        
        self.fashion_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
                            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        self.fashion_to_category = {
            'T-shirt/top': 't-shirt',
            'Trouser': 'pants',
            'Pullover': 'sweater',
            'Dress': 'dress',
            'Coat': 'coat',
            'Sandal': 'shoes',
            'Shirt': 'shirt',
            'Sneaker': 'sneakers',
            'Bag': 'handbag',
            'Ankle boot': 'boots'
        }
        
        self.style_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(STYLES))
        ).to(self.device)

        self.pattern_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(PATTERNS))
        ).to(self.device)
        
        print("Note: Using pre-trained fashion model and simulated style/pattern classifiers.", file=sys.stderr)
    
    def setup_logo_detection_model(self):
        """Initialize and load the logo detection model"""
        try:
            print("Loading logo detection model...", file=sys.stderr)
            
            self.logo_model = LogoClassificationModel(len(FASHION_BRANDS))
            
            try:
                self.logo_model.load_state_dict(torch.load("best_logo_classifier.pth", 
                                                          map_location=self.device))
                print("Successfully loaded logo detection model", file=sys.stderr)
            except Exception as e:
                print(f"Error loading logo detection model: {e}", file=sys.stderr)
                print("Logo detection will be simulated", file=sys.stderr)
            
            self.logo_model.to(self.device)
            self.logo_model.eval()
            
            self.logo_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            print(f"Error setting up logo detection: {e}", file=sys.stderr)
            self.logo_model = None
    
    def detect_logo(self, img):
        """Detect fashion brand logos in the image"""
        if self.logo_model is None:
            return None, 0.0
        
        try:
            img_tensor = self.logo_transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.logo_model(img_tensor)
                scores = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, class_idx = torch.max(scores, 0)
            
            brand_name = FASHION_BRANDS[class_idx.item()] if class_idx.item() < len(FASHION_BRANDS) else "Unknown"
            confidence_value = confidence.item()
            if confidence_value < 0.60:  
                return None, 0.0
                
            return brand_name, confidence_value
            
        except Exception as e:
            print(f"Error in logo detection: {e}", file=sys.stderr)
            return None, 0.0
    
    def preprocess_for_fashion_model(self, img):
        """Preprocess image for the fashion MNIST model - improved version"""
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(self.device)
    
    def read_image(self, image_path):
        """Read image with proper error handling for Unicode paths"""
        try:
            image = Image.open(image_path).convert('RGB')
            self.current_img = image  
            return image
        except Exception as e:
            print(f"PIL failed to load image: {e}", file=sys.stderr)
            
            try:
                cv_img = cv2.imread(image_path)
                if cv_img is None:
                    raise ValueError("OpenCV returned None")
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(cv_img)
                self.current_img = image 
                return image
            except Exception as e:
                print(f"OpenCV failed to load image: {e}", file=sys.stderr)
                
                try:
                    img_data = np.fromfile(image_path, dtype=np.uint8)
                    cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(cv_img)
                    self.current_img = image  
                    return image
                except Exception as e:
                    print(f"Binary read failed: {e}", file=sys.stderr)
                    raise IOError(f"Failed to load image using all methods: {e}")

    
    def extract_features(self, img):
        """Extract deep features from the image using our model"""
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)  
        with torch.no_grad():
            features = self.feature_model(img_tensor)  
            features = features.view(features.size(0), -1)  

        return features

    
    def analyze_content(self, features):
        """Analyze the clothing content using the pre-trained fashion classifier"""
        img = self.current_img  
        
        fashion_input = self.preprocess_for_fashion_model(img)
        
        with torch.no_grad():
            fashion_outputs = self.fashion_model(fashion_input)
            fashion_probs = torch.nn.functional.softmax(fashion_outputs, dim=1)
            fashion_indices = fashion_probs.argsort(dim=1, descending=True)[0]
        
        clothing_types = []
        for idx in fashion_indices[:2]: 
            fashion_class = self.fashion_classes[idx.item()]
            category = self.fashion_to_category.get(fashion_class, 't-shirt')  
            clothing_types.append(category)
        
        clothing_types = list(dict.fromkeys(clothing_types))
        
        if not clothing_types:
            clothing_types = ['t-shirt']

        style_logits = self.style_classifier(features)
        style_probs = torch.nn.functional.softmax(style_logits, dim=1)
        style_indices = style_probs.argsort(dim=1, descending=True)[0][:2]
        styles = [STYLES[idx.item()] for idx in style_indices]
        pattern_logits = self.pattern_classifier(features)
        pattern_probs = torch.nn.functional.softmax(pattern_logits, dim=1)
        pattern = PATTERNS[pattern_probs.argmax(dim=1).item()]

        return {
            "clothing_types": clothing_types,
            "styles": styles,
            "pattern": pattern
        }
    
    def extract_dominant_colors(self, img, num_colors=3):
        """Extract the dominant colors from the image using K-means clustering"""
        np_img = np.array(img)
        
        pixels = np_img.reshape(-1, 3)
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=num_colors, n_init=10)
            kmeans.fit(pixels)
            
            centers = kmeans.cluster_centers_
            labels = kmeans.labels_
            counts = Counter(labels)
            
            sorted_colors = [centers[i] for i, _ in counts.most_common()]
            return [tuple(map(int, color)) for color in sorted_colors]
        except ImportError:
            avg_color = np.mean(pixels, axis=0).astype(int)
            return [
                tuple(avg_color),
                tuple(np.clip(avg_color + 50, 0, 255).astype(int)),
                tuple(np.clip(avg_color - 50, 0, 255).astype(int))
            ]
    
    def name_color(self, rgb):
        """Find the closest named color for an RGB value"""
        if HAS_WEBCOLORS:
            try:
                closest_name = webcolors.rgb_to_name(rgb)
                return closest_name.lower()
            except (ValueError, AttributeError):
                pass
        
        min_diff = float('inf')
        closest_name = "unknown"
        
        for color_rgb, name in FALLBACK_COLORS.items():
            diff = sum(abs(v1 - v2) for v1, v2 in zip(color_rgb, rgb))
            
            if diff < min_diff:
                min_diff = diff
                closest_name = name
        
        return closest_name
    
    def suggest_material(self, clothing_type):
        """Suggest likely materials based on clothing type"""
        if clothing_type in ['jeans', 'denim jacket']:
            return 'denim'
        elif clothing_type in ['t-shirt', 'shirt']:
            return 'cotton'
        elif clothing_type in ['sweater', 'hoodie']:
            return 'fleece' if 'hoodie' in clothing_type else 'knit'
        elif clothing_type in ['dress', 'blouse']:
            return 'chiffon'
        elif clothing_type in ['coat', 'jacket']:
            return 'leather' if 'leather' in clothing_type else 'wool'
        elif clothing_type in ['shoes', 'boots']:
            return 'leather'
        else:
            import random
            return random.choice(MATERIALS)
    
    def suggest_season(self, clothing_type):
        """Suggest appropriate seasons based on clothing type"""
        if clothing_type in ['coat', 'sweater', 'hoodie']:
            return 'fall/winter'
        elif clothing_type in ['shorts', 'sandals']:
            return 'summer'
        elif clothing_type in ['t-shirt', 'light jacket']:
            return 'spring/summer'
        else:
            return 'all-season'
    
    def suggest_occasion(self, style):
        """Suggest occasions based on style"""
        if style in ['formal', 'business', 'professional']:
            return 'work or formal event'
        elif style in ['athletic', 'sporty']:
            return 'workout or casual outing'
        elif style in ['casual', 'streetwear', 'urban']:
            return 'everyday wear'
        elif style in ['elegant', 'chic']:
            return 'date night or special occasion'
        else:
            import random
            return random.choice(OCCASIONS)
    
    def generate_description_with_llm(self, analysis):
        """Generate a grounded, clean product description using a language model."""
        if not HAS_TRANSFORMERS or self.text_generator is None:
            return None

        try:
            clothing_type = analysis['clothing_types'][0]
            style = analysis['styles'][0]
            color = analysis['colors'][0]
            pattern = analysis['pattern']
            material = analysis['material']
            season = analysis['season']
            occasion = analysis['occasion']
            brand = analysis.get('brand', None)

            brand_text = f" from {brand}" if brand else ""

            prompt = (
                f"A {color} {pattern} {style} {clothing_type}{brand_text} made of {material}. "
                f"Perfect for {season}, ideal for {occasion}. Describe it in 2-3 elegant, natural sentences "
                f"without mentioning sizes, prices, or fabric percentages."
            )

            result = self.text_generator(
                prompt,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                max_length=180,
                pad_token_id=50256
            )

            description = result[0]['generated_text'].replace(prompt, '').strip()

            description = re.sub(r'#?[0-9a-fA-F]{6}', '', description)
            description = re.sub(r'\$\d+(\.\d+)?', '', description)
            description = re.sub(r'\b(size[s]?|XS|S|M|L|XL|XXL)\b', '', description, flags=re.IGNORECASE)

            sentences = re.split(r'(?<=[.!?])\s+', description)
            seen = set()
            filtered = []
            for s in sentences:
                s_clean = s.strip().lower()
                if s_clean in seen or len(s_clean) < 15:
                    continue
                seen.add(s_clean)
                filtered.append(s.strip())
            description = " ".join(filtered[:3])  

            if not description:
                return None

            return description

        except Exception as e:
            print(f"Error generating description with LLM: {e}", file=sys.stderr)
            return None
    
    def generate_template_description(self, analysis):
        """Generate a template-based description as fallback"""
        clothing_type = analysis['clothing_types'][0]
        style = analysis['styles'][0]
        color_names = analysis['colors']
        pattern = analysis['pattern']
        material = analysis['material']
        season = analysis['season']
        occasion = analysis['occasion']
        brand = analysis.get('brand', None)
        
        if len(color_names) > 1:
            color_str = f"{color_names[0]} and {color_names[1]}"
        else:
            color_str = color_names[0]
        
        brand_str = f" from {brand}" if brand else ""
        
        intro = f"Elevate your wardrobe with this {style} {clothing_type}{brand_str} in {color_str}."
        
        if pattern != 'solid':
            pattern_desc = f" The {pattern} pattern adds visual interest and contemporary flair."
        else:
            pattern_desc = f" The clean {pattern} design offers versatile styling options."
        
        material_desc = f" Crafted from premium {material} fabric that ensures comfort and durability."
        
        occasion_desc = f" Perfect for {occasion} during {season}."
        
        styling_tips = self.get_styling_tips(clothing_type, style, color_names[0])
        
        description = intro + pattern_desc + material_desc + occasion_desc + " " + styling_tips
        
        return description
    
    def get_styling_tips(self, clothing_type, style, color):
        """Generate styling tips based on the clothing attributes"""
        tops = ['top', 't-shirt', 'blouse', 'shirt', 'sweater', 'hoodie']
        bottoms = ['pants', 'jeans', 'shorts', 'skirt']
        outerwear = ['coat', 'jacket']
        footwear = ['shoes', 'sneakers', 'boots']
        accessories = ['hat', 'cap', 'scarf', 'tie', 'handbag', 'backpack', 'sunglasses', 'watch', 'jewelry', 'belt']
        
        if clothing_type in tops:
            return f"Style with your favorite {self.complementary_color_item(color, 'jeans')} for an effortless look that transitions from day to night."
        elif clothing_type in bottoms:
            return f"Pair with a {self.complementary_color_item(color, 'top')} to create a balanced silhouette that flatters your figure."
        elif clothing_type in outerwear:
            return f"Layer over a {style} outfit to add both warmth and sophisticated style to your ensemble."
        elif clothing_type in footwear:
            return f"These versatile {clothing_type} complement a wide range of outfits while providing all-day comfort."
        elif clothing_type in accessories:
            return f"Add this finishing touch to elevate even the simplest outfit into a fashion statement."
        else:
            return f"Mix and match with other pieces in your wardrobe for countless stylish combinations."
    
    def complementary_color_item(self, color, item_type):
        """Suggest a complementary colored item"""
        neutral_colors = ['black', 'white', 'gray', 'beige', 'navy']
        
        if color in neutral_colors:
            suggested_color = "contrasting"
        else:
            suggested_color = "neutral"
            
        return f"{suggested_color} {item_type}"
    
    def analyze_image(self, image_path):
        """Main function to analyze an image and generate a description"""
        try:
            print(f"Starting analysis for: {image_path}", file=sys.stderr)
            
            img = self.read_image(image_path)
            if img is None:
                print("Error: Failed to read image.", file=sys.stderr)
                return {
                    'analysis': {
                        'clothing_types': ['apparel item'],
                        'styles': ['stylish'],
                        'pattern': 'solid',
                        'colors': ['versatile'],
                        'material': 'premium',
                        'season': 'all-season',
                        'occasion': 'everyday'
                    },
                    'description': "A stylish, versatile apparel item crafted with attention to detail and premium materials. An excellent addition to any fashion-conscious wardrobe."
                }
            
            print("Detecting logos...", file=sys.stderr)
            brand, confidence = self.detect_logo(img)
            
            print("Extracting features...", file=sys.stderr)
            features = self.extract_features(img)
            
            print("Analyzing content...", file=sys.stderr)
            content_analysis = self.analyze_content(features)
            
            print("Analyzing colors...", file=sys.stderr)
            colors_rgb = self.extract_dominant_colors(img)
            color_names = [self.name_color(color) for color in colors_rgb]
            color_names = list(dict.fromkeys(color_names))  

            analysis = {
                'clothing_types': content_analysis['clothing_types'],
                'styles': content_analysis['styles'],
                'pattern': content_analysis['pattern'],
                'colors': color_names,
                'material': self.suggest_material(content_analysis['clothing_types'][0]),
                'season': self.suggest_season(content_analysis['clothing_types'][0]),
                'occasion': self.suggest_occasion(content_analysis['styles'][0])
            }
            
            if brand:
                analysis['brand'] = brand
                analysis['brand_confidence'] = f"{confidence:.2%}"
                print(f"Detected brand: {brand} (confidence: {confidence:.2%})", file=sys.stderr)
            
            print(f"Analysis results: {json.dumps(analysis, indent=2)}", file=sys.stderr)
            
            print("Generating description...", file=sys.stderr)
            description = self.generate_description_with_llm(analysis)
            
            if not description:
                print("Using fallback template description.", file=sys.stderr)
                description = self.generate_template_description(analysis)
            
            return {
                'analysis': analysis,
                'description': description
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}", file=sys.stderr)
            return {
                'analysis': {
                    'clothing_types': ['apparel item'],
                    'styles': ['stylish'],
                    'pattern': 'solid',
                    'colors': ['versatile'],
                    'material': 'premium',
                    'season': 'all-season',
                    'occasion': 'everyday'
                },
                'description': "A stylish, versatile apparel item crafted with attention to detail and premium materials. An excellent addition to any fashion-conscious wardrobe."
            }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze clothing images with deep learning.')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--output', choices=['description', 'json', 'both'], 
                       default='description', help='Output format')
    return parser.parse_args()


def clean_path(raw_path):
    """Normalize the file path, handle encoding issues, and preserve relative paths."""
    raw_path = raw_path.strip()
    
    if raw_path.startswith(("'", '"')) and raw_path.endswith(("'", '"')):
        raw_path = raw_path[1:-1]

    if raw_path.startswith('/uploads'):
        base_dir = os.getcwd()
        raw_path = raw_path[1:]
        return os.path.normpath(os.path.join(base_dir, raw_path))
    else:
        return os.path.normpath(os.path.abspath(raw_path))

def main():
    """Main function to process an image and create a description"""
    try:
        if len(sys.argv) < 2:
            print("Usage: python image_analyzer.py <path_to_image>", file=sys.stderr)
            sys.exit(1)

        image_path_raw = sys.argv[1]
        print(f"Raw path received: {image_path_raw!r}", file=sys.stderr)

        image_path = clean_path(image_path_raw)
        print(f"Cleaned path: {image_path!r}", file=sys.stderr)

        if not os.path.exists(image_path):
            print(f"Error: Image file does not exist: {image_path!r}", file=sys.stderr)
            print(f"Working directory: {os.getcwd()!r}", file=sys.stderr)
            print(f"File system encoding: {sys.getfilesystemencoding()}", file=sys.stderr)
            dir_path = os.path.dirname(image_path) or '.'
            
            print(f"Available files in directory {dir_path!r}:", file=sys.stderr)
            try:
                for f in os.listdir(dir_path):
                    print(f"  {f!r}", file=sys.stderr)
            except Exception as e:
                print(f"  Could not list directory: {e}", file=sys.stderr)

            base_name = os.path.basename(image_path)
            similar_files = []
            try:
                for f in os.listdir(dir_path):
                    if base_name.replace('-', '').lower() in f.replace('-', '').lower():
                        similar_files.append(f)
            except Exception:
                pass

            if similar_files:
                print("Similar files found:", file=sys.stderr)
                for f in similar_files:
                    print(f"  {f!r}", file=sys.stderr)

                image_path = os.path.join(dir_path, similar_files[0])
                print(f"Trying with similar file: {image_path!r}", file=sys.stderr)
            else:
                fallback_description = "A stylish, modern apparel item with unique design and high-quality materials. Perfect for adding versatility to your wardrobe."
                print("--- FALLBACK DESCRIPTION START ---", file=sys.stderr)
                print(fallback_description)  
                print("--- FALLBACK DESCRIPTION END ---", file=sys.stderr)
                sys.stdout.flush()
                sys.exit(1)

        print("Generating description...", file=sys.stderr)

        analyzer = ClothingImageAnalyzer()

        result = analyzer.analyze_image(image_path)
        description = result['description']

        print("--- DESCRIPTION START ---", file=sys.stderr)
        print(description)  
        print("--- DESCRIPTION END ---", file=sys.stderr)

        sys.stdout.flush()

    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)

        fallback_description = "A stylish, versatile apparel item crafted with attention to detail and premium materials. An excellent addition to any fashion-conscious wardrobe."
        print("--- FALLBACK DESCRIPTION START ---", file=sys.stderr)
        print(fallback_description)  
        print("--- FALLBACK DESCRIPTION END ---", file=sys.stderr)

if __name__ == "__main__":
    main()