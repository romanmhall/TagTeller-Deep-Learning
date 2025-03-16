# scraper_wikipedia.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Configure logger for this specific scraper
logger = logging.getLogger('scraper_wikipedia')

# Base URL for Wikipedia
BASE_URL = "https://en.wikipedia.org"
START_URL = "https://en.wikipedia.org/wiki/Lists_of_The_New_York_Times_number-one_books"


# Function to fetch page content
def get_soup(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
        else:
            logger.error(f"Failed to retrieve {url}, status code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


# Step 1: Extract year links
def get_year_links():
    logger.info(f"Fetching year links from {START_URL}")
    soup = get_soup(START_URL)
    if not soup:
        return []

    year_links = {}
    content_div = soup.find("div", class_="mw-parser-output")
    if content_div:
        for a in content_div.find_all("a", href=True):
            year = a.text.strip()
            if year.isdigit() and year not in year_links:  # Ensures only unique year links
                year_links[year] = BASE_URL + a["href"]

    logger.info(f"Found {len(year_links)} year links")
    return list(year_links.items())


# Step 2: Extract book data
def get_books_from_year(year, url):
    logger.info(f"Processing data for year {year} from {url}")
    soup = get_soup(url)
    if not soup:
        return []

    books = []
    tables = soup.find_all("table", class_="wikitable")
    if not tables:
        logger.warning(f"No tables found for {year}. Skipping...")
        return books

    for table in tables:
        category = "Fiction"
        heading = table.find_previous("h2")
        if heading and "Nonfiction" in heading.text:
            category = "Nonfiction"

        rows = table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 3:
                date = cols[0].get_text(strip=True)
                book_title = cols[1].get_text(strip=True)
                author = cols[2].get_text(strip=True)

                if "No List Published" not in book_title:
                    books.append((year, category, date, book_title, author))

    if not books:
        logger.warning(f"No valid book data found for {year}. Skipping...")
    else:
        logger.info(f"Extracted {len(books)} books from {year}.")

    return books


# Step 3: Scrape all books
def scrape_all_books():
    year_links = get_year_links()
    all_books = []

    for year, link in year_links:
        logger.info(f"Scraping {year}...")
        books = get_books_from_year(year, link)
        all_books.extend(books)
        time.sleep(1)  # Respectful scraping with delay

    return all_books


# This is the function our controller will call
def run_scraper():
    """
    Main function to run the scraper and return a pandas DataFrame.
    This is the entry point that will be called by the scraper controller.
    """
    logger.info("Starting Wikipedia NYT bestsellers scraper")
    try:
        books_data = scrape_all_books()

        if not books_data:
            logger.warning("No data scraped from Wikipedia. Check the page structure.")
            return pd.DataFrame()

        df = pd.DataFrame(books_data, columns=["Year", "Category", "Date", "Book Title", "Author"])
        logger.info(f"Successfully scraped {len(df)} books from Wikipedia")
        return df

    except Exception as e:
        logger.error(f"Error in Wikipedia scraper: {str(e)}")
        return pd.DataFrame()


# For standalone execution
if __name__ == "__main__":
    # Configure logging for standalone mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run the scraper and save to CSV directly
    df = run_scraper()
    if not df.empty:
        file_path = "nyt_best_sellers.csv"
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("No data scraped. Check the structure of the Wikipedia page.")