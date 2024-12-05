import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re


def download_file(url, local_path):
  response = requests.get(url)
  if response.status_code == 200:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, 'wb') as f:
      f.write(response.content)
    print(f"Downloaded: {local_path}")
  else:
    print(response.status_code)
    print(f"Failed to download: {url}")


def update_html(html_content, base_url):
  soup = BeautifulSoup(html_content, 'html.parser')

  for tag in soup.find_all(['script', 'link']):
    if tag.has_attr('src'):
      url = tag['src']
    elif tag.has_attr('href'):
      url = tag['href']
    else:
      continue

    if url.startswith(('http://', 'https://')):
      full_url = url
    else:
      full_url = urljoin(base_url, url)

    parsed_url = urlparse(full_url)
    local_path = os.path.join('static', parsed_url.netloc, parsed_url.path.lstrip('/'))

    download_file(full_url, local_path)

    relative_path = os.path.relpath(local_path, '.')
    if tag.name == 'script':
      tag['src'] = "/" + relative_path
    elif tag.name == 'link':
      tag['href'] = "/" + relative_path

  return str(soup)


# Read the HTML file
with open('./index.html', 'r') as f:
  html_content = f.read()

# Update HTML and download files
# updated_html = update_html(html_content, 'https://example.com')

# # Write the updated HTML
# with open('./index.html', 'w') as f:
#     f.write(updated_html)

print("HTML file updated with local paths.")

# Download Font Awesome CSS and font files
base_url = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/"
css_url = urljoin(base_url, "css/all.min.css")
output_dir = "static/cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2"

# Download CSS file
css_output_path = os.path.join(output_dir, "css", "all.min.css")
download_file(css_url, css_output_path)

# Parse CSS file for font URLs
with open(css_output_path, 'r', encoding='utf-8') as f:
  css_content = f.read()

# Extract font URLs from the CSS content
font_urls = re.findall(r'url\((.*?\.(?:woff2|ttf))\)', css_content)

print(f"Found {len(font_urls)} font URLs")

# Download font files
for font_url in font_urls:
  font_url = font_url.strip('"\'')
  if font_url.startswith('../'):
    font_url = font_url[3:]

  # Use base_url instead of urljoin to keep the version number
  full_url = base_url + font_url
  relative_path = font_url
  output_path = os.path.join(output_dir, relative_path)
  download_file(full_url, output_path)

print("Download complete!")
