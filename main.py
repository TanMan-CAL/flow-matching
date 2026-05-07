"""
Generate CFG diagram similar to the reference but with CIFAR-100 images.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_cifar_like_image(color_scheme, size=80):
    """Create a simple CIFAR-100 style image (e.g., airplane, car, etc)."""
    img = Image.new('RGB', (size, size), color='#87CEEB')  # sky blue
    draw = ImageDraw.Draw(img)
    
    if color_scheme == 'airplane':
        # Simple airplane shape
        draw.ellipse([25, 35, 55, 45], fill='#C0C0C0')  # fuselage
        draw.polygon([(15, 40), (30, 35), (30, 45)], fill='#C0C0C0')  # left wing
        draw.polygon([(50, 35), (65, 40), (50, 45)], fill='#C0C0C0')  # right wing
        draw.polygon([(55, 30), (60, 40), (55, 40)], fill='#C0C0C0')  # tail
    elif color_scheme == 'noisy':
        # Add noise
        pixels = np.array(img)
        noise = np.random.randint(-30, 30, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
    elif color_scheme == 'blurred':
        # Similar to airplane but blurrier
        from PIL import ImageFilter
        img_temp = Image.new('RGB', (size, size), color='#87CEEB')
        draw_temp = ImageDraw.Draw(img_temp)
        draw_temp.ellipse([25, 35, 55, 45], fill='#C0C0C0')
        draw_temp.polygon([(15, 40), (30, 35), (30, 45)], fill='#C0C0C0')
        draw_temp.polygon([(50, 35), (65, 40), (50, 45)], fill='#C0C0C0')
        draw_temp.polygon([(55, 30), (60, 40), (55, 40)], fill='#C0C0C0')
        img = img_temp.filter(ImageFilter.GaussianBlur(radius=2))
    
    return img

def create_cfg_diagram():
    """Create the CFG diagram."""
    # Canvas
    width, height = 1300, 800
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Create sample images
    x_t_clean = create_cifar_like_image('airplane')
    x_t_noisy = create_cifar_like_image('noisy')
    x_hat = create_cifar_like_image('blurred')
    
    # Positions
    left_x = 200
    right_x = 800
    top_y = 100
    
    # === LEFT SIDE: Classifier-free guidance ===
    
    # Title at top
    draw.text((left_x - 70, top_y - 30), "External\nCondition\nc", fill='black', font=font_label, anchor="mm")
    draw.line([(left_x - 70, top_y + 10), (left_x - 70, top_y + 80)], fill='purple', width=3)
    draw.polygon([(left_x - 70, top_y + 80), (left_x - 75, top_y + 70), (left_x - 65, top_y + 70)], fill='purple')
    
    # Top x_t image
    img.paste(x_t_noisy, (left_x - 40, top_y - 30))
    draw.text((left_x, top_y - 45), "x_t", fill='black', font=font_label, anchor="mm")
    
    # Dotted line from condition to x_t
    for i in range(0, 60, 8):
        draw.line([(left_x - 70 + i, top_y + 30), (left_x - 70 + i + 4, top_y + 30)], fill='orange', width=2)
    
    # Dotted line from x_t down to epsilon boxes
    for i in range(0, 150, 8):
        draw.line([(left_x, top_y + 50 + i), (left_x, top_y + 50 + i + 4)], fill='gray', width=2, dash=(4, 4))
    
    # Left epsilon box (conditional)
    box_size = 80
    draw.rectangle([left_x - 120, top_y + 210, left_x - 40, top_y + 290], outline='black', width=2, fill='white')
    draw.text((left_x - 80, top_y + 250), "ε_θ(·)", fill='black', font=font_label, anchor="mm")
    
    # Dotted connection from condition to left box
    for i in range(0, 40, 8):
        draw.line([(left_x - 80 - i, top_y + 80), (left_x - 80 - i + 4, top_y + 80)], fill='orange', width=2)
    for i in range(0, 130, 8):
        draw.line([(left_x - 120, top_y + 80 + i), (left_x - 120, top_y + 80 + i + 4)], fill='orange', width=2)
    
    # Right epsilon box (unconditional)  
    draw.rectangle([left_x + 40, top_y + 210, left_x + 120, top_y + 290], outline='black', width=2, fill='white')
    draw.text((left_x + 80, top_y + 250), "ε_θ(·)", fill='black', font=font_label, anchor="mm")
    
    # Text labels below boxes
    draw.text((left_x - 80, top_y + 305), "ε_θ(x_t, c)", fill='black', font=font_small, anchor="mm")
    draw.text((left_x, top_y + 320), "Trained\ndifferently", fill='black', font=font_small, anchor="mm")
    draw.text((left_x + 80, top_y + 305), "ε_θ(x_t)", fill='black', font=font_small, anchor="mm")
    
    # Arrows down from boxes
    draw.line([(left_x - 80, top_y + 290), (left_x - 80, top_y + 360)], fill='blue', width=2)
    draw.polygon([(left_x - 80, top_y + 360), (left_x - 85, top_y + 350), (left_x - 75, top_y + 350)], fill='blue')
    
    draw.line([(left_x + 80, top_y + 290), (left_x + 80, top_y + 360)], fill='blue', width=2)
    draw.polygon([(left_x + 80, top_y + 360), (left_x + 75, top_y + 350), (left_x + 85, top_y + 350)], fill='blue')
    
    # D_pos, Eq. 5, D_neg labels
    draw.text((left_x - 80, top_y + 370), "D_pos", fill='purple', font=font_small, anchor="mm")
    draw.text((left_x, top_y + 370), "Eq. 5", fill='black', font=font_small, anchor="mm")
    draw.text((left_x + 80, top_y + 370), "D_neg", fill='purple', font=font_small, anchor="mm")
    
    # Central output
    draw.text((left_x, top_y + 410), "ε̃(x_t)", fill='black', font=font_label, anchor="mm")
    draw.text((left_x, top_y + 450), "Eq. 1", fill='black', font=font_label, anchor="mm")
    draw.line([(left_x, top_y + 420), (left_x, top_y + 500)], fill='blue', width=2)
    draw.polygon([(left_x, top_y + 500), (left_x - 5, top_y + 490), (left_x + 5, top_y + 490)], fill='blue')
    
    # Bottom x_{t-1} image
    img.paste(x_t_clean, (left_x - 40, top_y + 510))
    draw.text((left_x, top_y + 605), "x_{t-1}", fill='black', font=font_label, anchor="mm")
    
    # Dashed feedback line (horizontal then vertical)
    for i in range(0, 250, 8):
        draw.line([(left_x + 40 + i, top_y + 555), (left_x + 40 + i + 4, top_y + 555)], fill='gray', width=2)
    for i in range(0, 555, 8):
        draw.line([(left_x + 290, top_y + i), (left_x + 290, top_y + i + 4)], fill='gray', width=2)
    
    draw.text((left_x + 120, top_y + 570), "D_out", fill='purple', font=font_small, anchor="mm")
    
    # === RIGHT SIDE: Self-attention guidance ===
    
    # Top x_t image
    img.paste(x_t_noisy, (right_x - 40, top_y - 30))
    draw.text((right_x, top_y - 45), "x_t", fill='black', font=font_label, anchor="mm")
    
    # Adversarial blurring box
    draw.rectangle([right_x + 50, top_y, right_x + 180, top_y + 60], outline='orange', width=2, fill='white')
    draw.text((right_x + 115, top_y + 25), "Adversarial\nBlurring", fill='orange', font=font_small, anchor="mm")
    
    # Dotted line from x_t to adversarial box
    for i in range(0, 90, 8):
        draw.line([(right_x + 40 + i, top_y + 15), (right_x + 40 + i + 4, top_y + 15)], fill='orange', width=2)
    
    # x_hat image
    img.paste(x_hat, (right_x - 40, top_y + 90))
    draw.text((right_x, top_y + 75), "x̂_t", fill='black', font=font_label, anchor="mm")
    
    # Dotted from adversarial to x_hat
    for i in range(0, 30, 8):
        draw.line([(right_x + 115, top_y + 60 + i), (right_x + 115, top_y + 60 + i + 4)], fill='orange', width=2)
    for i in range(0, 115, 8):
        draw.line([(right_x + i, top_y + 90), (right_x + i + 4, top_y + 90)], fill='orange', width=2)
    
    # M_t and A_t boxes
    m_t_img = Image.new('RGB', (50, 50), 'white')
    m_draw = ImageDraw.Draw(m_t_img)
    for i in range(0, 50, 10):
        for j in range(0, 50, 10):
            if (i + j) % 20 == 0:
                m_draw.rectangle([i, j, i+10, j+10], fill='black')
    img.paste(m_t_img, (right_x + 120, top_y + 210))
    
    a_t_img = Image.new('RGB', (50, 50), 'white')
    a_draw = ImageDraw.Draw(a_t_img)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    for i in range(0, 50, 10):
        for j in range(0, 50, 10):
            color = colors[(i//10 + j//10) % len(colors)]
            a_draw.rectangle([i, j, i+10, j+10], fill=color)
    img.paste(a_t_img, (right_x + 120, top_y + 290))
    draw.text((right_x + 145, top_y + 350), "A_t", fill='black', font=font_label, anchor="mm")
    draw.text((right_x + 230, top_y + 370), "Eq. 16", fill='black', font=font_small, anchor="mm")
    
    # Left epsilon box
    draw.rectangle([right_x - 120, top_y + 240, right_x - 40, top_y + 320], outline='black', width=2, fill='white')
    draw.text((right_x - 80, top_y + 280), "ε_θ(·)", fill='black', font=font_label, anchor="mm")
    
    # Dotted from x_hat to left epsilon
    for i in range(0, 80, 8):
        draw.line([(right_x - 80, top_y + 170 + i), (right_x - 80, top_y + 170 + i + 4)], fill='gray', width=2)
    
    # Dotted from M_t to left epsilon
    for i in range(0, 40, 8):
        draw.line([(right_x + 120 - i, top_y + 235), (right_x + 120 - i + 4, top_y + 235)], fill='orange', width=2)
    
    # Dotted from A_t to left epsilon  
    for i in range(0, 40, 8):
        draw.line([(right_x + 120 - i, top_y + 315), (right_x + 120 - i + 4, top_y + 315)], fill='orange', width=2)
    
    draw.text((right_x - 80, top_y + 335), "ε_θ(x̂_t)", fill='black', font=font_small, anchor="mm")
    
    # Right epsilon box
    draw.rectangle([right_x + 40, top_y + 240, right_x + 120, top_y + 320], outline='black', width=2, fill='white')
    draw.text((right_x + 80, top_y + 280), "ε_θ(·)", fill='black', font=font_label, anchor="mm")
    
    # Dotted from x_t to right epsilon
    for i in range(0, 200, 8):
        draw.line([(right_x, top_y + 50 + i), (right_x, top_y + 50 + i + 4)], fill='gray', width=2)
    for i in range(0, 80, 8):
        draw.line([(right_x + i, top_y + 250), (right_x + i + 4, top_y + 250)], fill='gray', width=2)
    
    draw.text((right_x + 80, top_y + 335), "ε_θ(x_t)", fill='black', font=font_small, anchor="mm")
    
    # Bottom arrows and labels
    draw.line([(right_x - 80, top_y + 320), (right_x - 80, top_y + 390)], fill='blue', width=2)
    draw.polygon([(right_x - 80, top_y + 390), (right_x - 85, top_y + 380), (right_x - 75, top_y + 380)], fill='blue')
    
    draw.line([(right_x + 80, top_y + 320), (right_x + 80, top_y + 390)], fill='blue', width=2)
    draw.polygon([(right_x + 80, top_y + 390), (right_x + 75, top_y + 380), (right_x + 85, top_y + 380)], fill='blue')
    
    draw.text((right_x - 80, top_y + 400), "D_neg", fill='purple', font=font_small, anchor="mm")
    draw.text((right_x + 80, top_y + 400), "D_pos", fill='purple', font=font_small, anchor="mm")
    
    # Central output
    draw.text((right_x, top_y + 440), "ε̃(x_t)", fill='black', font=font_label, anchor="mm")
    draw.text((right_x, top_y + 480), "Eq. 1", fill='black', font=font_label, anchor="mm")
    draw.line([(right_x, top_y + 450), (right_x, top_y + 530)], fill='blue', width=2)
    draw.polygon([(right_x, top_y + 530), (right_x - 5, top_y + 520), (right_x + 5, top_y + 520)], fill='blue')
    
    # Bottom x_{t-1} image
    img.paste(x_t_clean, (right_x - 40, top_y + 540))
    draw.text((right_x, top_y + 635), "x_{t-1}", fill='black', font=font_label, anchor="mm")
    
    # Dashed feedback
    for i in range(0, 300, 8):
        draw.line([(right_x + 40 + i, top_y + 585), (right_x + 40 + i + 4, top_y + 585)], fill='gray', width=2)
    for i in range(0, 585, 8):
        draw.line([(right_x + 340, top_y + i), (right_x + 340, top_y + i + 4)], fill='gray', width=2)
    
    draw.text((right_x + 170, top_y + 600), "D_out", fill='purple', font=font_small, anchor="mm")
    
    # Vertical separating line with "Next Step" labels
    mid_x = (left_x + right_x) // 2 + 100
    for i in range(50, height - 80, 15):
        draw.line([(mid_x, i), (mid_x, i + 8)], fill='lightgray', width=2)
    
    draw.text((mid_x, top_y + 150), "Next Step", fill='gray', font=font_label, anchor="mm")
    draw.text((mid_x, top_y + 420), "Next Step", fill='gray', font=font_label, anchor="mm")
    
    # Labels at bottom
    draw.text((left_x - 50, height - 60), "(a) Classifier-free guidance", fill='black', font=font_title, anchor="lm")
    draw.text((right_x - 100, height - 60), "(b) Self-attention guidance", fill='black', font=font_title, anchor="lm")
    
    return img

# Generate and save
os.makedirs('/home/claude/photos', exist_ok=True)
diagram = create_cfg_diagram()
diagram.save('/home/claude/photos/cfg_comparison_diagram.png')
print("Diagram saved to /home/claude/photos/cfg_comparison_diagram.png")