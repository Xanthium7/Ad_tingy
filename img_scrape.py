import requests
from bs4 import BeautifulSoup
import re


def search_images(query, num_images=10):

    query = query.replace(' ', '+')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Make a request to Bing images
    url = f"https://www.bing.com/images/search?q={query}"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Error: Request failed with status code {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    image_urls = []

    for img in soup.select('.mimg'):
        if img.has_attr('src') and img['src'].startswith('http'):
            image_urls.append(img['src'])
        elif img.has_attr('data-src') and img['data-src'].startswith('http'):
            image_urls.append(img['data-src'])

    # If we don't find enough images with the above method, try an alternative approach
    if len(image_urls) < num_images:
        # Look for image URLs in the page source
        img_tags = soup.find_all('img')
        for img in img_tags:
            if img.has_attr('src') and img['src'].startswith('http'):
                image_urls.append(img['src'])

    # Remove duplicates while preserving order
    image_urls = list(dict.fromkeys(image_urls))

    # Return the requested number of images (or all if fewer are found)
    return image_urls[:min(num_images, len(image_urls))]


def main():
    print("Image URL Scraper")
    print("-----------------")

    query = input("Enter a topic to search for images: ")

    print(f"\nSearching for images related to '{query}'...")

    image_urls = search_images(query)

    if image_urls:
        print(f"\nFound {len(image_urls)} image URLs:")
        for i, url in enumerate(image_urls, 1):
            print(f"{i}. {url}")
    else:
        print("No images found. Try a different search term.")


if __name__ == "__main__":
    main()
