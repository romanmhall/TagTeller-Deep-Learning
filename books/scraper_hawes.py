# scraper_hawes.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

# Configure logger for this specific scraper
logger = logging.getLogger('scraper_hawes')

# Define URLs for Fiction and Non-Fiction Best Sellers
FICTION_URL = "https://www.hawes.com/no1_f_t.htm"
NONFICTION_URL = "https://www.hawes.com/no1_nf_t.htm"


# Function to fetch HTML content
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


# Function to scrape book data
def scrape_books(url, category):
    logger.info(f"Scraping {category} books from {url}")
    soup = get_soup(url)
    if not soup:
        return []

    books = []

    # Find all <b> elements (since book title and author are inside)
    book_entries = soup.find_all("b")

    for entry in book_entries:
        try:
            # Extract book title from the <i> tag inside <b>
            title_tag = entry.find("i")
            book_title = title_tag.text.strip() if title_tag else "Unknown"

            # Extract the author by getting the next text after <b>
            author_text = entry.next_sibling  # Gets the text after <b>
            author = "Unknown"
            if author_text:
                author = author_text.strip()
                # Remove "by" prefix
                if "by " in author:
                    author = author.split("by ")[-1]
                # Remove the " -" at the end if it exists
                author = author.rstrip(" -")

            # Extract the date (inside <a> tag)
            date_tag = entry.find_next("a")
            date = date_tag.text.strip() if date_tag else "Unknown"

            # Add year extracted from date if available
            year = "Unknown"
            if date and len(date) >= 4:
                # Try to extract a year from the date string
                for i in range(len(date) - 3):
                    if date[i:i + 4].isdigit():
                        year = date[i:i + 4]
                        break

            # Append to list
            books.append((year, category, book_title, author, date))
        except Exception as e:
            logger.warning(f"Error parsing entry: {str(e)}")

    logger.info(f"Found {len(books)} {category} books")
    return books


# Main function for the controller to call
def run_scraper():
    """
    Main function to run the scraper and return a pandas DataFrame.
    This is the entry point that will be called by the scraper controller.
    """
    logger.info("Starting Hawes bestsellers scraper")
    try:
        # Scrape fiction and non-fiction books
        fiction_books = scrape_books(FICTION_URL, "Fiction")
        nonfiction_books = scrape_books(NONFICTION_URL, "Non-Fiction")

        # Combine both categories
        all_books = fiction_books + nonfiction_books

        if not all_books:
            logger.warning("No data scraped from Hawes. Check the page structure.")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(all_books, columns=["Year", "Category", "Book Title", "Author", "Date"])
        logger.info(f"Successfully scraped {len(df)} books from Hawes")
        return df

    except Exception as e:
        logger.error(f"Error in Hawes scraper: {str(e)}")
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
        file_path = "hawes_best_sellers.csv"
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("No data scraped. Check the website structure.")