# scraper_wikifiction.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging

# Configure logger for this specific scraper
logger = logging.getLogger('scraper_wikifiction')

# Wikipedia URL for best-selling fiction authors
WIKI_URL = 'https://en.wikipedia.org/wiki/List_of_best-selling_fiction_authors'


# Function to fetch and parse the Wikipedia page
def fetch_soup(url):
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


# Function to extract author information
def extract_authors(soup):
    authors = []
    logger.info("Extracting best-selling fiction authors from Wikipedia")

    # Locate the main table containing the authors
    table = soup.find('table', {'class': 'wikitable'})

    if table:
        # Iterate over each row, skipping the header row
        for row in table.find_all('tr')[1:]:
            try:
                columns = row.find_all('td')

                if columns:
                    # Extract author name from the first column, which contains an <a> tag
                    author_tag = columns[0].find('a')
                    author_name = author_tag.text.strip() if author_tag else columns[0].text.strip()

                    # Extract approximate sales if available
                    sales = columns[1].text.strip() if len(columns) > 1 else "Unknown"

                    # Extract author nationality if available
                    nationality = columns[2].text.strip() if len(columns) > 2 else "Unknown"

                    # Extract language if available
                    language = columns[3].text.strip() if len(columns) > 3 else "Unknown"

                    # Extract notes if available
                    notes = columns[4].text.strip() if len(columns) > 4 else ""

                    # Use a consistent format for the data
                    # We'll use "Best-Selling Fiction Authors" as a category
                    # and include additional information in notes
                    authors.append(("Current", "Best-Selling Fiction Authors", author_name,
                                    f"Sales: {sales}, Nationality: {nationality}, Language: {language}", notes))
            except Exception as e:
                logger.warning(f"Error parsing row: {str(e)}")
    else:
        logger.warning("No author table found on the page.")

    logger.info(f"Found {len(authors)} best-selling fiction authors")
    return authors


# Main function for the controller to call
def run_scraper():
    """
    Main function to run the scraper and return a pandas DataFrame.
    This is the entry point that will be called by the scraper controller.
    """
    logger.info("Starting Wikipedia best-selling fiction authors scraper")
    try:
        soup = fetch_soup(WIKI_URL)
        if not soup:
            logger.error("Failed to retrieve data from Wikipedia")
            return pd.DataFrame()

        authors = extract_authors(soup)
        if not authors:
            logger.warning("No authors found on Wikipedia page")
            return pd.DataFrame()

        # Create DataFrame with consistent column names
        df = pd.DataFrame(authors, columns=["Year", "Category", "Author", "Book Title", "Notes"])
        logger.info(f"Successfully scraped {len(df)} best-selling fiction authors")
        return df

    except Exception as e:
        logger.error(f"Error in Wikipedia fiction authors scraper: {str(e)}")
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
        file_path = "wiki_bestselling_authors.csv"
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print("No data scraped. Check the Wikipedia page structure.")