import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re

def process_line(line):
    # Remove the prefix "-" or "- " or "number. " or "number."
    line_prefix = ""
    if re.match(r"^(-\s|-(?=\S))", line):
        line_prefix = "• "
        line = re.sub(r"^(-\s|-(?=\S))", "", line)
    elif re.match(r"^(\d+\.)\s", line):
        line_prefix = re.search(r"^(\d+\.)\s", line).group(0)
        line = re.sub(r"^(\d+\.)\s", "", line)
    elif re.match(r"^(\d+\.)", line):
        line_prefix = re.search(r"^(\d+\.)", line).group(0) + " "
        line = re.sub(r"^(\d+\.)", "", line)
    
    # Remove the part after ":"
    if ":" in line:
        line = line.split(":")[0]
    
    return line_prefix + line.strip()

def process_input(input_string):
    name_list = []
    lines = input_string.strip().split('\n')
    for line in lines:
        processed = process_line(line)
        if processed:
            name_list.append(processed)
    
    # Check if the name_list contains fewer than 3 elements
    if len(name_list) < 3:
        name_list = []

    return name_list

def overlay_images(background_img_path, overlay_img_path, output_path, position=(0, 0), text_list=None, font_path='MADEOkineSansPERSONALUSE-Bold.otf', title=None, text=None):
    if text_list is None:
        text_list = []

    text_list = [line.replace("-", "•", 1) for line in text_list]

    # Load images
    background_img = cv2.imread(background_img_path)
    overlay_img = cv2.imread(overlay_img_path, -1)

    if len(text_list) > 0:
        background_img = cv2.GaussianBlur(background_img, (17, 17), 0)
    background_img = cv2.resize(background_img, (1640, 840))
    background_height, background_width, _ = background_img.shape

    # Resize overlay image to match the size of the background image
    overlay_img = cv2.resize(overlay_img, (background_img.shape[1], background_img.shape[0]))

    # Get position to overlay the image
    y_start, x_start = position
    y_end, x_end = y_start + overlay_img.shape[0], x_start + overlay_img.shape[1]

    # Ensure the overlay does not exceed the dimensions of the background image
    y_end = min(y_end, background_img.shape[0])
    x_end = min(x_end, background_img.shape[1])

    # Overlay the image on the background
    overlay = background_img[y_start:y_end, x_start:x_end]

    # Blend the images
    for c in range(0, 3):
        overlay[:, :, c] = overlay[:, :, c] * (1 - overlay_img[:, :, 3] / 255.0) + \
                            overlay_img[:, :, c] * (overlay_img[:, :, 3] / 255.0)

    # Update the background image with the overlay
    background_img[y_start:y_end, x_start:x_end] = overlay

    # Convert the image to PIL format
    pil_img = Image.fromarray(cv2.cvtColor(background_img, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # Create a transparent image for text drawing
    text_layer = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)

    # Load the custom font
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)

    # Define text properties
    line_spacing = 15

    font_size_text = 54
    font_text = ImageFont.truetype(font_path, font_size_text)
    line_spacing_text = 10
    max_line_width = 1520

    text = text.upper()
    lines = []
    words = text.split()

    while words:
        line = ''
        while words and draw.textbbox((0, 0), line + words[0], font=font_text)[2] <= max_line_width:
            line += words.pop(0) + ' '
        lines.append(line)

    if len(text) > 55:
        partition_index = text.find(":")
        if partition_index != -1:
            first_line = text[:partition_index + 1]
            second_line = text[partition_index + 1:].strip()
        else:
            first_line = text
            second_line = ''

        total_text_height = draw.textbbox((0, 0), first_line, font=font_text)[3] + (draw.textbbox((0, 0), second_line, font=font_text)[3] + line_spacing_text if second_line else 0)
        text_y = background_img.shape[0] - 70 - total_text_height
        first_line_width = draw.textbbox((0, 0), first_line, font=font_text)[2]
        text_x = (background_img.shape[1] - first_line_width) // 2
        draw.text((text_x, text_y), first_line, font=font_text, fill=(255, 255, 255))

        if second_line:
            text_y += draw.textbbox((0, 0), first_line, font=font_text)[3] + line_spacing_text
            second_line_width = draw.textbbox((0, 0), second_line, font=font_text)[2]
            text_x = (background_img.shape[1] - second_line_width) // 2
            draw.text((text_x, text_y), second_line, font=font_text, fill=(255, 255, 255))
    else:
        text_y = background_img.shape[0] - 90 - len(lines) * (draw.textbbox((0, 0), text, font=font_text)[3] + line_spacing_text)
        for line in lines:
            line_width = draw.textbbox((0, 0), line, font=font_text)[2]
            text_x = (background_img.shape[1] - line_width) // 2
            draw.text((text_x, text_y), line, font=font_text, fill=(255, 255, 255))
            text_y += draw.textbbox((0, 0), line, font=font_text)[3] + line_spacing_text

    def draw_text_with_border(draw, position, text, font, fill, border_color, border_width):
        x, y = position
        for dx in range(-border_width, border_width + 1):
            for dy in range(-border_width, border_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=border_color)
        draw.text(position, text, font=font, fill=fill)

    white_transparent = (255, 255, 255, 255)
    black = (0, 0, 0, 200)
    if len(text_list) > 0:
        if len(text_list) <= 8:
            col1_text = text_list
            col1_x = (background_img.shape[1] - max([draw.textbbox((0, 0), line, font=font)[2] for line in col1_text])) // 2
            col1_y = 110
            for line in col1_text:
                draw_text_with_border(draw, (col1_x, col1_y), line, font, white_transparent, black, 2)
                col1_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing
        elif len(text_list) <= 16:
            mid_point = (len(text_list) + 1) // 2
            col1_text = text_list[:mid_point]
            col2_text = text_list[mid_point:]
            col1_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in col1_text])
            col2_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in col2_text])
            col1_x = (background_img.shape[1] - col1_width - col2_width - 80) // 2
            col2_x = col1_x + col1_width + 200
            col1_y = 110
            col2_y = 110
            for line in col1_text:
                draw_text_with_border(draw, (col1_x, col1_y), line, font, white_transparent, black, 2)
                col1_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing
            for line in col2_text:
                draw_text_with_border(draw, (col2_x, col2_y), line, font, white_transparent, black, 2)
                col2_y += draw.textbbox((0, 0), line, font=font)[3] + 17

        else:
            # Three columns
            col_len = (len(text_list) + 2) // 3
            col1_text = text_list[:col_len]
            col2_text = text_list[col_len:2*col_len]
            col3_text = text_list[2*col_len:]
            col1_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in col1_text])
            col2_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in col2_text])
            col3_width = max([draw.textbbox((0, 0), line, font=font)[2] for line in col3_text])
            col1_x = (background_width - col1_width - col2_width - 210) // 2
            col2_x = col1_x + col1_width + 20
            col3_x = col2_x + col2_width + 20
            col1_y = 110
            col2_y = 110
            col3_y = 110
            for line in col1_text:
                draw_text_with_border(draw, (col1_x, col1_y), line, font, white_transparent, black, 2)
                col1_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing
            for line in col2_text:
                draw_text_with_border(draw, (col2_x, col2_y), line, font, white_transparent, black, 2)
                col2_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing
            for line in col3_text:
                draw_text_with_border(draw, (col3_x, col3_y), line, font, white_transparent, black, 2)
                col3_y += draw.textbbox((0, 0), line, font=font)[3] + line_spacing

    # Composite the text layer onto the original image
    pil_img = Image.alpha_composite(pil_img, text_layer)
    
    # Convert back to OpenCV format
    background_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    if title != "":
        title = title.lower()
        colon_index = title.find(':')
        if colon_index != -1:
            title = title[:colon_index]
        else:
            title = title
        title = ' '.join(title.split())
    
        translation_table = str.maketrans('', '', "!.;?/[]{}()#@^&*")

        # Use translate to remove the specified characters
        title = title.translate(translation_table)
        title = title.replace(" ", "_")
        title += "_1640x840.png"
        output_path = title

    cv2.imwrite(output_path, background_img)

# Example usage
background_image_path = 'pool.png'
overlay_image_path = 'frame.png'
title = "Black algae: Identification, Repair, Causes, Preventions"
name_list = []

input_string = """ """

name_list = process_input(input_string)

output_image_path = 'output_image.jpg'
overlay_position = (0, 0)  # Adjust position as needed
font_path = 'MADEOkineSansPERSONALUSE-Bold.otf'

overlay_images(background_image_path, overlay_image_path, output_image_path, overlay_position, text_list=name_list, font_path=font_path, title=title, text=title)
