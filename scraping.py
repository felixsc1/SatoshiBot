import requests
from bs4 import BeautifulSoup
import os
import time
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SatoshiScraper:
    def __init__(self, base_url="https://satoshi.nakamotoinstitute.org", output_dir="nakamotoinstitute_files"):
        self.base_url = base_url
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def sanitize_filename(self, filename):
        """Sanitize filename to be safe for filesystem"""
        # Remove or replace problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        # Limit length
        if len(filename) > 100:
            filename = filename[:100]
        return filename
    
    def extract_post_content(self, soup):
        """Extract relevant content from forum posts"""
        content_parts = []
        
        # Try to find the main post content
        post_div = soup.find('div', class_='post')
        if post_div:
            content_parts.append(f"<div class='post-content'>{post_div.get_text(strip=True)}</div>")
        
        # Also try to extract header information (title, date, etc.)
        title_elem = soup.find('h1')
        if title_elem:
            content_parts.append(f"<h1>{title_elem.get_text(strip=True)}</h1>")
        
        # Extract date if available
        time_elem = soup.find('time')
        if time_elem:
            datetime_attr = time_elem.get('dateTime', '')
            time_text = time_elem.get_text(strip=True)
            content_parts.append(f"<div class='date' datetime='{datetime_attr}'>{time_text}</div>")
        
        # Extract category/source info
        h2_elem = soup.find('h2', class_='small-caps')
        if h2_elem:
            content_parts.append(f"<div class='category'>{h2_elem.get_text(strip=True)}</div>")
        
        # Fallback: if no specific content found, try to extract from main content area
        if not content_parts:
            main_elem = soup.find('main')
            if main_elem:
                # Remove navigation and footer elements
                for nav in main_elem.find_all(['nav', 'footer']):
                    nav.decompose()
                content_parts.append(f"<div class='main-content'>{main_elem.get_text(strip=True)}</div>")
        
        return '\n'.join(content_parts) if content_parts else ""
    
    def extract_email_content(self, soup):
        """Extract relevant content from emails"""
        content_parts = []
        
        # Extract email header information
        header_section = soup.find('header', class_='border-taupe border-b border-dashed font-mono')
        if header_section:
            # Extract From, Subject, Date
            grid_div = header_section.find('div', class_='grid')
            if grid_div:
                header_text = grid_div.get_text(separator=' | ', strip=True)
                content_parts.append(f"<div class='email-header'>{header_text}</div>")
        
        # Extract main email content using multiple strategies
        email_content = None
        
        # Strategy 1: Look for section with px-8 py-4 classes (and possibly more classes)
        email_sections = soup.find_all('section')
        for section in email_sections:
            classes = section.get('class', [])
            if 'px-8' in classes and 'py-4' in classes:
                email_content = section
                break
        
        # Strategy 2: CSS selector approach
        if not email_content:
            email_content = soup.select_one('section.px-8.py-4')
        
        # Strategy 3: Look for any section with font-mono that contains substantial text
        if not email_content:
            font_mono_sections = soup.find_all('section', class_=lambda x: x and 'font-mono' in x)
            for section in font_mono_sections:
                text = section.get_text(strip=True)
                if text and len(text) > 50:  # Ensure it has substantial content
                    email_content = section
                    break
        
        # Strategy 4: Look for div inside main content that contains the email body
        if not email_content:
            main_elem = soup.find('main')
            if main_elem:
                # Look for divs that contain substantial email-like content
                divs = main_elem.find_all('div')
                for div in divs:
                    text = div.get_text(strip=True)
                    if text and len(text) > 100:  # Substantial content
                        # Check if it looks like email content (contains common email patterns)
                        if any(pattern in text.lower() for pattern in ['wrote:', 'from:', 'subject:', '-----', 'unsubscribe', 'mailing list']):
                            email_content = div
                            break
        
        if email_content:
            # Get the text content, preserving some structure
            content_text = email_content.get_text(separator='\n', strip=True)
            content_parts.append(f"<div class='email-content'>{content_text}</div>")
        
        # Extract title if available
        title_elem = soup.find('h1')
        if title_elem:
            content_parts.append(f"<h1>{title_elem.get_text(strip=True)}</h1>")
        
        # Extract category/source info
        h2_elem = soup.find('h2', class_='small-caps')
        if h2_elem:
            content_parts.append(f"<div class='category'>{h2_elem.get_text(strip=True)}</div>")
        
        # Enhanced fallback: if no specific content found, try to extract from main content area
        if len(content_parts) <= 2:  # Only header and title, no actual content
            main_elem = soup.find('main')
            if main_elem:
                # Create a copy to avoid modifying the original
                main_copy = main_elem.__copy__()
                
                # Remove navigation and footer elements
                for elem in main_copy.find_all(['nav', 'footer', 'header']):
                    elem.decompose()
                
                # Get all text content
                main_text = main_copy.get_text(separator='\n', strip=True)
                
                # Look for substantial content after removing headers
                lines = main_text.split('\n')
                content_lines = []
                for line in lines:
                    line = line.strip()
                    if line and len(line) > 10:  # Skip short lines
                        content_lines.append(line)
                
                if content_lines:
                    content_parts.append(f"<div class='main-content'>{'<br/>'.join(content_lines)}</div>")
        
        return '\n'.join(content_parts) if content_parts else ""
    
    def extract_quote_content(self, soup):
        """Extract relevant content from quotes"""
        content_parts = []
        
        # Extract title
        title_elem = soup.find('h1')
        if title_elem:
            content_parts.append(f"<h1>{title_elem.get_text(strip=True)}</h1>")
        
        # Extract main quote content - look for various possible containers
        quote_selectors = [
            'div.quote-content',
            'section.px-8',
            'div.prose',
            'blockquote',
            'main section'
        ]
        
        quote_found = False
        for selector in quote_selectors:
            quote_elem = soup.select_one(selector)
            if quote_elem:
                quote_text = quote_elem.get_text(separator='\n', strip=True)
                if quote_text and len(quote_text) > 50:  # Ensure it's substantial content
                    content_parts.append(f"<div class='quote-content'>{quote_text}</div>")
                    quote_found = True
                    break
        
        # Extract source/date information
        time_elem = soup.find('time')
        if time_elem:
            datetime_attr = time_elem.get('dateTime', '')
            time_text = time_elem.get_text(strip=True)
            content_parts.append(f"<div class='date' datetime='{datetime_attr}'>{time_text}</div>")
        
        # Fallback: if no specific content found, try to extract from main content area
        if not quote_found:
            main_elem = soup.find('main')
            if main_elem:
                # Remove navigation and footer elements
                for nav in main_elem.find_all(['nav', 'footer']):
                    nav.decompose()
                main_text = main_elem.get_text(strip=True)
                if main_text:
                    content_parts.append(f"<div class='main-content'>{main_text}</div>")
        
        return '\n'.join(content_parts) if content_parts else ""
    
    def download_and_extract_content(self, url, filename, content_type='post'):
        """Download a page and extract only relevant content"""
        try:
            logger.info(f"Downloading: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse the HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content based on type
            if content_type == 'post':
                extracted_content = self.extract_post_content(soup)
            elif content_type == 'email':
                extracted_content = self.extract_email_content(soup)
            elif content_type == 'quote':
                extracted_content = self.extract_quote_content(soup)
            else:
                # Default extraction
                extracted_content = self.extract_post_content(soup)
            
            if not extracted_content:
                logger.warning(f"No content extracted from {url}")
                return False
            
            # Create a clean HTML document with just the extracted content
            clean_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Extracted Content</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .email-header, .post-header {{ background-color: #f5f5f5; padding: 10px; margin-bottom: 20px; }}
        .email-content, .post-content, .quote-content {{ margin-bottom: 20px; }}
        .date {{ color: #666; font-size: 0.9em; }}
        .category {{ font-weight: bold; color: #333; }}
        h1 {{ color: #2c3e50; }}
    </style>
</head>
<body>
    <div class="source-url">Source: <a href="{url}">{url}</a></div>
    {extracted_content}
</body>
</html>"""
            
            # Save the clean HTML
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(clean_html)
            
            logger.info(f"Saved clean content: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download and extract {url}: {str(e)}")
            return False
    
    def scrape_emails(self):
        """Scrape all email pages"""
        logger.info("Starting to scrape emails...")
        
        # Get the main emails page
        emails_url = f"{self.base_url}/emails/"
        try:
            response = self.session.get(emails_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all email links
            email_links = []
            # Look for links that contain email-like patterns
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/emails/' in href and href != '/emails/':
                    full_url = urljoin(self.base_url, href)
                    email_links.append((full_url, link.get_text(strip=True)))
            
            logger.info(f"Found {len(email_links)} email links")
            
            # Create emails subdirectory
            emails_dir = os.path.join(self.output_dir, 'emails')
            os.makedirs(emails_dir, exist_ok=True)
            
            # Download each email
            for i, (url, title) in enumerate(email_links, 1):
                filename = f"email_{i:03d}_{self.sanitize_filename(title)}.html"
                filepath = os.path.join('emails', filename)
                self.download_and_extract_content(url, filepath, content_type='email')
                time.sleep(1)  # Be respectful to the server
                
        except Exception as e:
            logger.error(f"Error scraping emails: {str(e)}")
    
    def scrape_posts(self):
        """Scrape all forum posts"""
        logger.info("Starting to scrape posts...")
        
        # Get the main posts page
        posts_url = f"{self.base_url}/posts/"
        try:
            response = self.session.get(posts_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all post links
            post_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/posts/' in href and href != '/posts/':
                    full_url = urljoin(self.base_url, href)
                    post_links.append((full_url, link.get_text(strip=True)))
            
            logger.info(f"Found {len(post_links)} post links")
            
            # Create posts subdirectory
            posts_dir = os.path.join(self.output_dir, 'posts')
            os.makedirs(posts_dir, exist_ok=True)
            
            # Download each post
            for i, (url, title) in enumerate(post_links, 1):
                filename = f"post_{i:03d}_{self.sanitize_filename(title)}.html"
                filepath = os.path.join('posts', filename)
                self.download_and_extract_content(url, filepath, content_type='post')
                time.sleep(1)  # Be respectful to the server
                
        except Exception as e:
            logger.error(f"Error scraping posts: {str(e)}")
    
    def scrape_quotes(self):
        """Scrape all quotes"""
        logger.info("Starting to scrape quotes...")
        
        # Get the main quotes page
        quotes_url = f"{self.base_url}/quotes/"
        try:
            response = self.session.get(quotes_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all quote links
            quote_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/quotes/' in href and href != '/quotes/':
                    full_url = urljoin(self.base_url, href)
                    quote_links.append((full_url, link.get_text(strip=True)))
            
            logger.info(f"Found {len(quote_links)} quote links")
            
            # Create quotes subdirectory
            quotes_dir = os.path.join(self.output_dir, 'quotes')
            os.makedirs(quotes_dir, exist_ok=True)
            
            # Download each quote
            for i, (url, title) in enumerate(quote_links, 1):
                filename = f"quote_{i:03d}_{self.sanitize_filename(title)}.html"
                filepath = os.path.join('quotes', filename)
                self.download_and_extract_content(url, filepath, content_type='quote')
                time.sleep(1)  # Be respectful to the server
                
        except Exception as e:
            logger.error(f"Error scraping quotes: {str(e)}")
    
    def scrape_all(self):
        """Scrape all sections"""
        logger.info("Starting comprehensive scraping of Satoshi Nakamoto Institute...")
        
        # Download main pages first (keeping full HTML for these)
        self.download_and_extract_content(f"{self.base_url}/emails/", "emails_main.html", content_type='email')
        self.download_and_extract_content(f"{self.base_url}/posts/", "posts_main.html", content_type='post')
        self.download_and_extract_content(f"{self.base_url}/quotes/", "quotes_main.html", content_type='quote')
        
        # Scrape individual sections
        self.scrape_emails()
        self.scrape_posts()
        self.scrape_quotes()
        
        logger.info("Scraping completed!")

def main():
    """Main function to run the scraper"""
    scraper = SatoshiScraper()
    
    print("Enhanced Satoshi Nakamoto Institute Scraper")
    print("=" * 50)
    print("This will scrape and extract clean content from:")
    print("1. All emails from /emails/ (header + content only)")
    print("2. All posts from /posts/ (post content only)")
    print("3. All quotes from /quotes/ (quote content only)")
    print(f"Files will be saved to: {scraper.output_dir}")
    print("\nContent extraction features:")
    print("- Removes navigation, footers, and boilerplate")
    print("- Preserves essential content and metadata")
    print("- Creates clean, readable HTML files")
    print()
    
    choice = input("Do you want to proceed? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        scraper.scrape_all()
    else:
        print("Scraping cancelled.")

if __name__ == "__main__":
    main()
