import logging
import os
import sys
import time
from urllib.parse import urljoin, urlparse, parse_qs

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger('arize.download_docs')

sys.setrecursionlimit(1500)

"""Downloads the documentation from a given URL and saves it to a directory."""


class DocumentationDownloader:

    def __init__(self, configs):
        self.sleep_time = configs.sleep_time
        self.base_url = configs.website
        self.output_dir = configs.docs_dir
        self.visited_urls = set()
        self.session = requests.Session()
        self.max_depth = configs.max_depth
        self.no_of_docs = configs.no_of_docs
        self.downloaded_docs = 0
        self.language_code = configs.language
        self.index_file = "index.html"
        self.configs = configs
        self.file_names = list()

    def download_documentation(self):
        logger.info(f"Downloading documentation from {self.base_url}")
        self._recursive_download(self.base_url, 0)
        logger.info(f"Downloaded {self.downloaded_docs} docs")

    def extract_language_code(self, url):
        if self.configs.language_query_param:
            parsed_url = urlparse(url)
            query_params = parse_qs(parsed_url.query)
            language_code = query_params.get(self.configs.language_query_param)
            return language_code[0][:2] if language_code else 'en'
        else:
            language = url.replace("https://", '').split("/")
            if len(language) > 2 and len(language[1]) == 2:
                return language[1]
            elif len(language) > 2 and  '-' in language[1] and len(language[1].split('-')[0]) == 2:
                return language[1].split('-')[0]
            elif len(language) > 2 and  '_' in language[1] and len(language[1].split('-')[0]) == 2:
                return language[1].split('-')[0]
        return 'en'

    def _recursive_download(self, url, depth):
        language = self.extract_language_code(url)
        if language != self.language_code:
            return
        if url in self.visited_urls or url.split("#")[0] in self.visited_urls:
            # This is part of the already visited page, it represents the heading of the page.
            return
        if 0 < self.no_of_docs < self.downloaded_docs:
            return
        self.visited_urls.add(url.split("#")[0])
        try:
            filename = self._url_to_filename(url)
            filepath = os.path.join(self.output_dir, filename)
            
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
            else:
                if self.sleep_time:
                    time.sleep(self.sleep_time)
                logger.info(f"Downloading URL: {url}")
                response = self.session.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')

            # Check if any paragraph has more than 15 words
            has_long_paragraph = False
            for p in soup.find_all('p'):
                if len(p.get_text().split()) > 15:
                    has_long_paragraph = True
                    break

            if has_long_paragraph:
                if filename not in self.file_names:
                    self.file_names.append(filename)
                    self.downloaded_docs += 1
                    # Save the current page only if it has long paragraphs
                    if not os.path.isfile(filepath):
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(response.text)

                # Find and process all links
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    # Only follow links within the same domain
                    if self._is_same_domain(full_url) and depth + 1 < self.max_depth:
                        self._recursive_download(full_url, depth + 1)

        except requests.RequestException as e:
            if '404 Client Error' in str(e):
                logger.info(f"Page not found: {url}")
                return
            logger.error(f"Error downloading {url}: {e}")
            self.session = requests.Session()

    def _url_to_filename(self, url):
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        if not path:
            return self.index_file
        file_path = path.replace('/', '_').split(".html")[0]
        return f'{file_path}.html'

    def _is_same_domain(self, url):
        return urlparse(url).netloc == urlparse(self.base_url).netloc
