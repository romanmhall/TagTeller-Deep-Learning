# scraper_wikibooks.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

# Configure logger for this specific scraper
logger = logging.getLogger('scraper_wikibooks')

# Wikipedia URL for best-selling books
WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_best-selling_books'


# Function to fetch HTML content
def get_soup(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
        else:
            logger.error(f"Failed to retrieve {url}, Status Code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {str(e)}")
        return None


# Function to scrape book data from Wikipedia
def scrape_best_selling_books():
    logger.info(f"Scraping best-selling books from {WIKI_URL}")
    soup = get_soup(WIKI_URL)
    if not soup:
        return []

    books_data = []
    category = None  # To store the current category (e.g., "More than 100 million copies")

    # Find all tables in the page
    tables = soup.find_all("table", class_="wikitable")

    # Find section headers for categorization
    section_headers = soup.find_all("h3")
    section_index = 0  # Keep track of which section we're in

    for table in tables:
        try:
            # If a new section header appears, update the category
            if section_index < len(section_headers):
                category = section_headers[section_index].get_text(strip=True)
                # Remove the [edit] text often found in Wikipedia section headers
                category = category.replace("[edit]", "").strip()
                section_index += 1  # Move to the next section
                logger.info(f"Processing section: {category}")

            rows = table.find_all("tr")[1:]  # Skip header row

            for row in rows:
                try:
                    cols = row.find_all("td")

                    # Ensure we have at least 4 columns (Book Title, Author, First Published, Genre)
                    if len(cols) >= 4:
                        book_title = cols[0].get_text(strip=True) if len(cols) > 0 else "Unknown"
                        author = cols[1].get_text(strip=True) if len(cols) > 1 else "Unknown"
                        published_year = cols[3].get_text(strip=True) if len(cols) > 3 else "Unknown"
                        genre = cols[5].get_text(strip=True) if len(cols) > 5 else "Unknown"  # Handles missing genre

                        # Extract a clean year from the published_year field
                        year = "Unknown"
                        if published_year and len(published_year) >= 4:
                            # Try to extract a 4-digit year
                            for i in range(len(published_year) - 3):
                                if published_year[i:i + 4].isdigit():
                                    year = published_year[i:i + 4]
                                    break

                        books_data.append((year, category, book_title, author, genre))
                except Exception as e:
                    logger.warning(f"Error parsing row: {str(e)}")
        except Exception as e:
            logger.warning(f"Error processing table: {str(e)}")

    logger.info(f"Found {len(books_data)} best-selling books")
    return books_data


# Main function for the controller to call
def run_scraper():
    """
    Main function to run the scraper and return a pandas DataFrame.
    This is the entry point that will be called by the scraper controller.
    """
    logger.info("Starting Wikipedia best-selling books scraper")
    try:
        books = scrape_best_selling_books()

        if not books:
            logger.warning("No books found from Wikipedia. Check the page structure.")
            return pd.DataFrame()

        # Create DataFrame with consistent column names matching other scrapers
        df = pd.DataFrame(books, columns=["Year", "Category", "Book Title", "Author", "Genre"])
        logger.info(f"Successfully scraped {len(df)} best-selling books")
        return df

    except Exception as e:
        logger.error(f"Error in Wikipedia best-selling books scraper: {str(e)}")
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
        file_path = "wiki_bestselling_books.csv"
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("No data scraped. Check the Wikipedia page structure.")