import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def download_file(url, local_path):
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {local_path}")
    else:
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
            tag['src'] = relative_path
        elif tag.name == 'link':
            tag['href'] = relative_path

    return str(soup)

# Read the HTML file
with open('./index.html', 'r') as f:
    html_content = f.read()

# Update HTML and download files
updated_html = update_html(html_content, 'https://example.com')

# Write the updated HTML
with open('./index.html', 'w') as f:
    f.write(updated_html)

print("HTML file updated with local paths.")