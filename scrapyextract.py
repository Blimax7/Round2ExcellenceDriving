import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# Initialize a set to keep track of visited URLs
visited_urls = set()

def scrape_website(url):
    # Avoid scraping the same URL multiple times
    if url in visited_urls:
        return
    visited_urls.add(url)

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')

        # Scrape sections separately (headers and paragraphs)
        sections = soup.find_all(['h1', 'h2', 'h3', 'p'])  # Scrape headers and paragraphs
        text_content = [section.get_text() for section in sections if section.get_text().strip()]  # Ensure non-empty text
        
        # Create a directory for storing scraped content if it doesn't exist
        output_dir = 'scraped_data'
        os.makedirs(output_dir, exist_ok=True)

        # Save the scraped content to a text file named after the page
        page_name = urlparse(url).path.replace('/', '_') or 'home'
        output_file_path = os.path.join(output_dir, f'{page_name}.txt')
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(text_content))

        print(f"Scraped content saved to {output_file_path}")

        # Find all links on the page and scrape them recursively
        for link in soup.find_all('a', href=True):
            full_url = urljoin(url, link['href'])
            if urlparse(full_url).netloc == urlparse(url).netloc:  # Only follow internal links
                scrape_website(full_url)  # Recursively scrape linked pages

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while scraping {url}: {e}")

# Start scraping from the main website URL
website_url = 'https://www.excellencedriving.com'
scrape_website(website_url)
