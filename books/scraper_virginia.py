# scraper_virginia.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Configure logger for this specific scraper
logger = logging.getLogger('scraper_virginia')

# Base URL for the Virginia Library site
VIRGINIA_START_URL = "https://bestsellers.lib.virginia.edu/"


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


# Step 1: Extract decade links from Virginia Library
def get_decade_links():
    logger.info(f"Fetching decade links from {VIRGINIA_START_URL}")
    soup = get_soup(VIRGINIA_START_URL)
    if not soup:
        return []

    decade_links = []
    for a in soup.find_all("a", href=True):
        if any(decade in a.text for decade in ["1900-1909", "1910-1919", "1920-1929", "1930-1939", "1940-1949",
                                               "1950-1959", "1960-1969", "1970-1979", "1980-1989", "1990-1999"]):
            decade_links.append((a.text.strip(), VIRGINIA_START_URL + a["href"]))

    logger.info(f"Found {len(decade_links)} decade links")
    return decade_links


# Step 2: Scrape book data from each decade page
def get_virginia_books():
    decade_links = get_decade_links()
    all_books = []

    for decade, link in decade_links:
        logger.info(f"Scraping {decade}...")
        decade_soup = get_soup(link)

        if not decade_soup:
            continue

        current_year = None
        tables = decade_soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 1:
                    year_text = cols[0].get_text(strip=True)
                    if ':' in year_text:
                        current_year = year_text.split(':')[0]  # Extract year from "1901: Fiction"
                elif len(cols) >= 3 and current_year:
                    try:
                        rank = cols[1].get_text(strip=True)
                        book_title = cols[2].get_text(strip=True)
                        author = cols[3].get_text(strip=True) if len(cols) > 3 else "Unknown"

                        if book_title:
                            all_books.append((current_year, "Fiction", rank, book_title, author))
                    except Exception as e:
                        logger.warning(f"Error parsing row in {decade}: {str(e)}")

        # Be respectful with scraping
        time.sleep(1)

    logger.info(f"Extracted {len(all_books)} books from Virginia Library")
    return all_books


# Main function that the controller will call
def run_scraper():
    """
    Main function to run the scraper and return a pandas DataFrame.
    This is the entry point that will be called by the scraper controller.
    """
    logger.info("Starting Virginia Library bestsellers scraper")
    try:
        books_data = get_virginia_books()

        if not books_data:
            logger.warning("No data scraped from Virginia Library. Check the page structure.")
            return pd.DataFrame()

        df = pd.DataFrame(books_data, columns=["Year", "Category", "Rank", "Book Title", "Author"])
        logger.info(f"Successfully scraped {len(df)} books from Virginia Library")
        return df

    except Exception as e:
        logger.error(f"Error in Virginia Library scraper: {str(e)}")
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
        file_path = "virginia_best_sellers.csv"
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("No data scraped. Check the structure of the Virginia Library page.")