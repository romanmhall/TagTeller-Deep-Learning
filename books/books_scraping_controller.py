import importlib.util
import sys
import pandas as pd
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"books_scraper_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('books_scraper_controller')


class BooksScraperController:
    def __init__(self, output_file="combined_books_data.csv", scrapers=None):
        """
        Initialize the books scraper controller.

        Args:
            output_file (str): Path to the output CSV file.
            scrapers (list): List of scraper module file paths (.py files).
        """
        self.output_file = output_file
        self.scrapers = scrapers or []
        logger.info(f"Initialized BooksScraperController with output file: {output_file}")

    def add_scraper(self, scraper_path):
        """Add a scraper module path to the controller."""
        if not scraper_path.endswith('.py'):
            scraper_path += '.py'

        if not os.path.exists(scraper_path):
            logger.error(f"Scraper file not found: {scraper_path}")
            return False

        self.scrapers.append(scraper_path)
        logger.info(f"Added scraper: {scraper_path}")
        return True

    def import_module_from_file(self, module_path):
        """
        Import a module from a file path dynamically.
        """
        try:
            module_name = os.path.basename(module_path).replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Error importing module {module_path}: {str(e)}")
            return None

    def run_all_scrapers(self):
        """Run all scrapers and combine their results."""
        logger.info(f"Starting to run {len(self.scrapers)} scrapers")

        all_data = []

        for scraper_path in self.scrapers:
            try:
                # Attempt to import the scraper module from file path
                logger.info(f"Importing scraper: {scraper_path}")
                scraper = self.import_module_from_file(scraper_path)

                if scraper is None:
                    continue

                # Check if the module has a run_scraper function
                if hasattr(scraper, 'run_scraper'):
                    logger.info(f"Running scraper: {scraper_path}")
                    data = scraper.run_scraper()

                    # Verify that the scraper returned data
                    if data is not None and not data.empty:
                        # Add a source column to identify which scraper produced this data
                        module_name = os.path.basename(scraper_path).replace('.py', '')
                        data['source'] = module_name
                        all_data.append(data)
                        logger.info(f"Successfully obtained data from {module_name}: {len(data)} rows")
                    else:
                        logger.warning(f"Scraper {scraper_path} returned empty data")
                else:
                    logger.error(f"Scraper {scraper_path} does not have a run_scraper function")
            except Exception as e:
                logger.error(f"Error running scraper {scraper_path}: {str(e)}")

        # Combine all dataframes if we have any
        if all_data:
            try:
                # Try to handle different column structures
                combined_df = pd.concat(all_data, ignore_index=True, sort=False)

                # Ensure all required columns exist
                required_columns = ["Year", "Category", "Book Title", "Author"]
                for col in required_columns:
                    if col not in combined_df.columns:
                        combined_df[col] = "Unknown"

                # Drop the columns we don't need: Rank, Notes, Genre
                columns_to_drop = ["Rank", "Notes", "Genre"]
                for col in columns_to_drop:
                    if col in combined_df.columns:
                        combined_df = combined_df.drop(columns=[col])
                        logger.info(f"Dropped column: {col}")

                # Check if the output file exists to determine if we need to write headers
                file_exists = os.path.isfile(self.output_file)

                # Ensure columns are in the preferred order
                preferred_columns = ["Year", "Category", "Date", "Book Title", "Author", "source"]
                # Filter to only include columns that exist in the dataframe
                available_preferred_columns = [col for col in preferred_columns if col in combined_df.columns]
                # Add any other columns that might exist
                other_columns = [col for col in combined_df.columns if col not in preferred_columns]
                ordered_columns = available_preferred_columns + other_columns

                # Reorder columns
                combined_df = combined_df[ordered_columns]

                # Write to CSV
                logger.info(f"Writing {len(combined_df)} rows to {self.output_file}")
                combined_df.to_csv(self.output_file, mode='a' if file_exists else 'w',
                                   header=not file_exists, index=False)

                logger.info("Successfully completed the scraping operation")
                return combined_df
            except Exception as e:
                logger.error(f"Error combining data: {str(e)}")
                return pd.DataFrame()
        else:
            logger.warning("No data was collected from any of the scrapers")
            return pd.DataFrame()

    def generate_report(self, data=None):
        """Generate a summary report of the scraped data."""
        if data is None:
            # Try to read the existing CSV file
            if os.path.exists(self.output_file):
                data = pd.read_csv(self.output_file)
            else:
                logger.error("No data provided and no output file exists")
                return None

        report = {}

        # Total number of entries
        report["total_entries"] = len(data)

        # Count by source
        if "source" in data.columns:
            report["entries_by_source"] = data["source"].value_counts().to_dict()

        # Count by category
        if "Category" in data.columns:
            report["entries_by_category"] = data["Category"].value_counts().to_dict()

        # Most frequent authors
        if "Author" in data.columns:
            top_authors = data["Author"].value_counts().head(10).to_dict()
            report["top_authors"] = top_authors

        # Distribution by year (if available)
        if "Year" in data.columns:
            # Filter out non-year values
            valid_years = data[data["Year"].str.isdigit()]
            if not valid_years.empty:
                years_count = valid_years["Year"].value_counts().to_dict()
                report["entries_by_year"] = {k: years_count[k] for k in sorted(years_count.keys())}

        logger.info("Report generated successfully")
        return report


# Example usage
if __name__ == "__main__":
    controller = BooksScraperController()

    # Add all five book scrapers (with actual file paths)
    controller.add_scraper("scraper_wikipedia.py")
    controller.add_scraper("scraper_virginia.py")
    controller.add_scraper("scraper_hawes.py")
    controller.add_scraper("scraper_wikifiction.py")
    controller.add_scraper("scraper_wikibooks.py")

    # Run all scrapers and get combined data
    result = controller.run_all_scrapers()

    # Generate a report
    if not result.empty:
        report = controller.generate_report(result)
        print("\n=== SCRAPING SUMMARY ===")
        print(f"Total entries collected: {report['total_entries']}")

        if "entries_by_source" in report:
            print("\nEntries by source:")
            for source, count in report["entries_by_source"].items():
                print(f"  - {source}: {count}")

        if "entries_by_category" in report:
            print("\nTop categories:")
            for category, count in sorted(report["entries_by_category"].items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  - {category}: {count}")

        if "top_authors" in report:
            print("\nTop authors:")
            for author, count in report["top_authors"].items():
                print(f"  - {author}: {count}")

        print(f"\nData saved to: {controller.output_file}")
    else:
        print("No data was collected. Check the logs for errors.")