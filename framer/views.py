# views.py
from django.shortcuts import render
from .forms import ImageUploadForm
import cv2
import numpy as np
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.contrib.staticfiles import finders
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import re
import os
import requests

background_img_path = ""
overlay_img_path = ""
current_output_path = ""
frame_image_path = ""
output_image_path = 'output_image.png'
overlay_position = (0, 0)  # Adjust position as needed

target_width = 1640
target_height = 840
min_width = 1440
max_width = 1840

name_list = []
title = "Black algae: Identification, Repair, Causes, Preventions"

input_string = """

1. 18 types of pools: Inground, Above Ground, Infinity
2. 4 types of inground pools: Vinyl, Fiberglass, Concrete, Gunite
3.Inground vinyl liner pool: Definition, Types, Cost
4.Inground fiber glass pool: Definition, Shapes, Cost
5. Concrete Pools: Definition, Shapes, Cost
1. 18 types of pools: Inground, Above Ground, Infinity
2. 4 types of inground pools: Vinyl, Fiberglass, Concrete, Gunite
3.Inground vinyl liner pool: Definition, Types, Cost
4.Inground fiber glass pool: Definition, Shapes, Cost
5. Concrete Pools: Definition, Shapes, Cost
-18 types of pools: Inground, Above Ground, Infinity
-4 types of inground pools: Vinyl, Fiberglass, Concrete, Gunite
- Inground vinyl liner pool: Definition, Types, Cost
- Inground fiber glass pool: Definition, Shapes, Cost
-Concrete Pools: Definition, Shapes, Cost



"""




output_image_path = 'output_image.jpg'
overlay_position = (0, 0)  # Adjust position as needed

font_path = 'MADEOkineSansPERSONALUSE-Bold.otf'

def process_line(line):
    # Remove the prefix "-" or "- " or "number. " or "number."
    line_prefix = ""
    if re.match(r"^-?\s*\d+\.\s*", line):
        line_prefix = re.search(r"^-?\s*\d+\.\s*", line).group(0)
        line = re.sub(r"^-?\s*\d+\.\s*", "", line)
    elif re.match(r"^-", line):
        line_prefix = "• "
        line = re.sub(r"^-", "", line)
    
    # Remove the part after ":"
    if ":" in line:
        line = line.split(":")[0]
    
    return line_prefix + line.strip() if line_prefix else None

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

name_list = process_input(input_string)

def transparent(overlay_img_path, output_path, position=(0, 0), text_list=None, font_path='MADEOkineSansPERSONALUSE-Bold.otf', title=None, text=None):
    global current_output_path
    if text_list is None:
        text_list = []

    text_list = [line.replace("-", "•", 1) for line in text_list]

    pil_img = Image.open(overlay_img_path)
    background_img = cv2.imread(overlay_img_path)
    background_height, background_width, _ = background_img.shape
    

    # Create a transparent image for text drawing
    text_layer = Image.new("RGBA", pil_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)

    # Load the custom font
    font_size = 40
    font_path = finders.find('fonts/MADEOkineSansPERSONALUSE-Bold.otf')
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

    
    


    current_output_path = output_path
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
        title += ".png"
        output_path = title
        current_output_path = output_path
    
    pil_img = Image.alpha_composite(pil_img, text_layer)
    background_img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    pil_img.save(output_path)

    cv2.imwrite(output_path, background_img)
    pil_img.save("output_image.png")

    return background_img

def get_dominant_color(image_path):
    # Read the image using cv2
    image = cv2.imread(image_path)
    # Convert image to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))
    # Convert to float for kmeans
    pixels = np.float32(pixels)
    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 1  # Number of clusters (dominant color)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Get the dominant color
    dominant_color = centers[0].astype(int)
    return tuple(dominant_color)

def overlay_images(background_img_path, overlay_img_path, output_path, position=(0, 0), text_list=None, font_path='MADEOkineSansPERSONALUSE-Bold.otf', title=None, text=None):
    global current_output_path
    if text_list is None:
        text_list = []

    text_list = [line.replace("-", "•", 1) for line in text_list]


    

    
    img = Image.open(background_img_path)
    img_width, img_height = img.size

    blur_img = img.resize((1640, 840), Image.LANCZOS)
    blur_img = np.array(blur_img)
    blur_img = Image.fromarray(blur_img)

    # Scale the image to match the target height
    new_height = target_height
    new_width = int(img_width * (target_height / img_height))
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    new_width, new_height = resized_img.size
    print(resized_img.size)

    if new_width < 1439:
        # Stretch the width to min_width and recalculate height
        new_width = 1439
        new_height = 840
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    if min_width <= new_width <= max_width:
        # Stretch the image to fit the target dimensions if within the width range
        stretched_img = resized_img.resize((target_width, target_height), Image.LANCZOS)
        new_img = stretched_img
    else:
        # Get the dominant color of the image
        offset = ((blur_img.width - resized_img.width) // 2, (blur_img.height - resized_img.height) // 2)

        # Paste the resized image on top of the blurred background
        new_img = blur_img.copy()
        new_img.paste(resized_img, offset)



    background_img = cv2.cvtColor(np.array(new_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    overlay_img = cv2.imread(overlay_img_path, -1)

    if len(text_list) > 0:
        pass
        

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
    font_path = finders.find('fonts/MADEOkineSansPERSONALUSE-Bold.otf')
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

    current_output_path = output_path
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
        title += ".png"
        output_path = title
        current_output_path = output_path

    cv2.imwrite(output_path, background_img)
    cv2.imwrite("output_image.png", background_img)
    
    

    return background_img

def get_pexels_images(api_key, query, per_page=100, page=1):
    url = "https://api.pexels.com/v1/search"
    headers = {
        "Authorization": api_key
    }
    params = {
        "query": query,
        "per_page": per_page,
        "page": page
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        photos = data.get('photos', [])
        return photos
    else:
        print(f"Failed to fetch images: {response.status_code} - {response.text}")
        return []

# Main view
def image_upload_view(request):
    global current_output_path, frame_image_path
    api_key = "563492ad6f9170000100000159be2064dd894ec2add4d0e635cc60b2"

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        title = request.POST.get('title', '')
        input_string = request.POST.get('input_string', '')

        # Handle search functionality
        if 'search' in request.POST:
            
            title = request.POST.get('title', '')
            input_string = request.POST.get('input_string', '')
            query = request.POST.get('search_query', '')
            photos = get_pexels_images(api_key, query)
            with open("temp_frame_image.png", 'rb') as f:
                image_file = SimpleUploadedFile(name='temp_frame_image.png', content=f.read(), content_type='image/png')
            form = ImageUploadForm(initial={'frame_image': image_file})
            
            return render(request, 'image_upload.html', {'form': form, 'photos': photos, 'title': title,
                'input_string': input_string, 'frame_url': '/static/temp_frame_image.png',})

        # Handle image selection from Pexels
        pexels_image_url = request.POST.get('pexels_image_url')
        pool_image = None
        if pexels_image_url:
            response = requests.get(pexels_image_url)
            if response.status_code == 200:
                pool_image = np.asarray(bytearray(response.content), dtype="uint8")
                pool_image = cv2.imdecode(pool_image, cv2.IMREAD_UNCHANGED)

        # Handle frame image URL from static files
        frame_image_url = request.POST.get('frame_image_url')
        frame_image = None
        if frame_image_url:
            response = requests.get(request.build_absolute_uri(frame_image_url))
            if response.status_code == 200:
                frame_image = np.asarray(bytearray(response.content), dtype="uint8")
                frame_image = cv2.imdecode(frame_image, cv2.IMREAD_UNCHANGED)
        else:
            pass
            

        # Process input for image processing
        name_list = process_input(input_string)

        if form.is_valid():
            frame_image_file = request.FILES.get('frame_image')
            title = request.POST.get('title')
            input_string = request.POST.get('input_string')
            if frame_image_file:
                frame_image = cv2.imdecode(np.frombuffer(frame_image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            # If no pool image is selected, use the uploaded file
            if pool_image is None:
                pool_image_file = request.FILES.get('pool_image')
                if pool_image_file:
                    pool_image = cv2.imdecode(np.frombuffer(pool_image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

            if frame_image is not None:
                frame_image_path = 'temp_frame_image.png'
                cv2.imwrite(frame_image_path, frame_image)

            if pool_image is None:
                # Only frame image provided
                output_image_path = "output_image.png"
                output_image = transparent(overlay_img_path=frame_image_path, output_path=output_image_path,
                                           position=overlay_position, text_list=name_list, font_path=font_path, title=title, text=title)
            else:
                # Both frame and pool images provided
                pool_image_path = 'temp_pool_image.png'
                frame_image_path = 'temp_frame_image.png'
                output_image_path = "output_image.png"
                cv2.imwrite(pool_image_path, pool_image)

                output_image = overlay_images(background_img_path=pool_image_path, overlay_img_path=frame_image_path, output_path=output_image_path,
                                              position=overlay_position, text_list=name_list, font_path=font_path, title=title, text=title)

            # Convert the output image to base64 to pass to the template
            with open(output_image_path, "rb") as img_file:
                output_image_data = base64.b64encode(img_file.read()).decode('utf-8')

            context = {
                'form': form,
                'output_image_data': output_image_data,
                'current_output_path': current_output_path,
                'title': title,
                'input_string': input_string,
                'frame_url': '/static/temp_frame_image.png',
            }
            return render(request, 'image_upload.html', context)
    else:
        form = ImageUploadForm()
    
    return render(request, 'image_upload.html', {'form': form})






def home(request):
    return render(request, "home.html")



def success(request):
    return render(request, 'success.html')
