from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import numpy as np
import cv2

def draw_rounded_rectangle(draw, coords, radius, fill):
  left, top, right, bottom = coords
  diameter = radius * 2
  draw.rectangle([left + radius, top, right - radius, bottom], fill=fill)
  draw.rectangle([left, top + radius, right, bottom - radius], fill=fill)
  draw.pieslice([left, top, left + diameter, top + diameter], 180, 270, fill=fill)
  draw.pieslice([right - diameter, top, right, top + diameter], 270, 360, fill=fill)
  draw.pieslice([left, bottom - diameter, left + diameter, bottom], 90, 180, fill=fill)
  draw.pieslice([right - diameter, bottom - diameter, right, bottom], 0, 90, fill=fill)

def draw_centered_text_rounded(draw, text, font, rect_coords, radius=10, text_color="yellow", bg_color=(43,33,44)):
  bbox = font.getbbox(text)
  text_width = bbox[2] - bbox[0]
  text_height = bbox[3] - bbox[1]
  rect_left, rect_top, rect_right, rect_bottom = rect_coords
  rect_width = rect_right - rect_left
  rect_height = rect_bottom - rect_top
  text_x = rect_left + (rect_width - text_width) // 2
  text_y = rect_top + (rect_height - text_height) // 2
  draw_rounded_rectangle(draw, rect_coords, radius, bg_color)
  draw.text((text_x, text_y), text, fill=text_color, font=font)

def draw_left_aligned_text_rounded(draw, text, font, rect_coords, padding_left=20, radius=10, text_color="yellow", bg_color=(43,33,44)):
  bbox = font.getbbox(text)
  text_height = bbox[3] - bbox[1]
  rect_left, rect_top, rect_right, rect_bottom = rect_coords
  rect_height = rect_bottom - rect_top
  text_y = rect_top + (rect_height - text_height) // 2
  text_x = rect_left + padding_left
  draw_rounded_rectangle(draw, rect_coords, radius, bg_color)
  draw.text((text_x, text_y), text, fill=text_color, font=font)

def draw_right_text_dynamic_width_rounded(draw, text, font, base_coords, padding=20, radius=10, text_color="yellow", bg_color=(43,33,44)):
  bbox = font.getbbox(text)
  text_width = bbox[2] - bbox[0]
  text_height = bbox[3] - bbox[1]
  _, rect_top, rect_right, rect_bottom = base_coords
  rect_height = rect_bottom - rect_top
  new_rect_left = rect_right - (text_width + (padding * 2))
  text_y = rect_top + (rect_height - text_height) // 2
  text_x = new_rect_left + padding
  draw_rounded_rectangle(draw, (new_rect_left, rect_top, rect_right, rect_bottom), radius, bg_color)
  draw.text((text_x, text_y), text, fill=text_color, font=font)
  return new_rect_left

def draw_progress_bar(draw, progress, coords, color="yellow", bg_color=(70, 70, 70)):
  left, top, right, bottom = coords
  total_width = right - left
  draw.rectangle(coords, fill=bg_color)
  progress_width = int(total_width * progress)
  if progress_width > 0:
    draw.rectangle((left, top, left + progress_width, bottom), fill=color)

def crop_image(image, top_crop=70):
  width, height = image.size
  return image.crop((0, top_crop, width, height))

def create_animation_mp4(
  replacement_image_path,
  output_path,
  device_name,
  prompt_text,
  fps=30,
  target_size=(512, 512),
  target_position=(139, 755),
  progress_coords=(139, 1285, 655, 1295),
  device_coords=(1240, 370, 1640, 416),
  prompt_coords=(332, 1702, 2662, 1745)
):
  frames = []
  try:
    font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 20)
    promptfont = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 24)
  except:
    font = ImageFont.load_default()
    promptfont = ImageFont.load_default()

  # Process first two frames
  for i in range(1, 3):
    base_img = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", f"image{i}.png"))
    draw = ImageDraw.Draw(base_img)
    draw_centered_text_rounded(draw, device_name, font, device_coords)
    if i == 2:
      draw_left_aligned_text_rounded(draw, prompt_text, promptfont, prompt_coords)
    frames.extend([crop_image(base_img)] * 30)  # 1 second at 30fps

  # Create blur sequence
  replacement_img = Image.open(replacement_image_path)
  base_img = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", "image3.png"))
  blur_steps = [int(80 * (1 - i/8)) for i in range(9)]

  for i, blur_amount in enumerate(blur_steps):
    new_frame = base_img.copy()
    draw = ImageDraw.Draw(new_frame)

    replacement_copy = replacement_img.copy()
    replacement_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
    if blur_amount > 0:
      replacement_copy = replacement_copy.filter(ImageFilter.GaussianBlur(radius=blur_amount))

    mask = replacement_copy.split()[-1] if replacement_copy.mode in ('RGBA', 'LA') else None
    new_frame.paste(replacement_copy, target_position, mask)

    draw_progress_bar(draw, (i + 1) / 9, progress_coords)
    draw_centered_text_rounded(draw, device_name, font, device_coords)
    draw_right_text_dynamic_width_rounded(draw, prompt_text, promptfont, (None, 590, 2850, 685), padding=30)

    frames.extend([crop_image(new_frame)] * 15)  # 0.5 seconds at 30fps

  # Create and add final frame (image4)
  final_base = Image.open(os.path.join(os.path.dirname(__file__), "baseimages", "image4.png"))
  draw = ImageDraw.Draw(final_base)

  draw_centered_text_rounded(draw, device_name, font, device_coords)
  draw_right_text_dynamic_width_rounded(draw, prompt_text, promptfont, (None, 590, 2850, 685), padding=30)

  replacement_copy = replacement_img.copy()
  replacement_copy.thumbnail(target_size, Image.Resampling.LANCZOS)
  mask = replacement_copy.split()[-1] if replacement_copy.mode in ('RGBA', 'LA') else None
  final_base.paste(replacement_copy, target_position, mask)

  frames.extend([crop_image(final_base)] * 30)  # 1 second at 30fps

  # Convert frames to video
  if frames:
    first_frame = np.array(frames[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
      out.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video saved successfully to {output_path}")
