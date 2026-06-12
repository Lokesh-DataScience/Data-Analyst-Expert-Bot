from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException
import json, re, os, textwrap, warnings

warnings.filterwarnings('ignore')

# Automatically download and use ChromeDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 10)

def wait_and_get(by, value, multiple=False):
    """ 
    Waits for an element to be present in the DOM and returns it.
    If multiple is True, returns a list of elements.
    """
    try:
        if multiple:
            return wait.until(EC.presence_of_all_elements_located((by, value))) # Returns a list of elements
        else:
            return wait.until(EC.presence_of_element_located((by, value))) # Returns a single element
    except TimeoutException:
        return [] if multiple else None # Returns None

def extract_content():
    """ 
    Extracts the main content from the article page.
    Returns a list of text content.
    """
    content = [] 
    try:
        container = driver.find_element(By.TAG_NAME, "td")
        for tag in ['p', 'ol', 'ul']: # Tags to extract content from
            for element in container.find_elements(By.TAG_NAME, tag): # Find elements by tag
                for sub in element.find_elements(By.TAG_NAME, "li") if tag in ['ol', 'ul'] else [element]: # Find sub-elements
                    text = sub.text.strip()
                    if text:
                        content.append(text)
    except NoSuchElementException:
        pass
    return content

def get_pagination_buttons():
    """ 
    Retrieves pagination buttons from the modal results.
    Returns a list of pagination button elements.
    """
    try:
        return driver.find_elements(By.CSS_SELECTOR, "span.pagination-number") # Returns a list of pagination buttons
    except Exception:
        return []

def scrape_details(context: str):
    """
    Scrapes articles from TPointTech based on the provided context.
    Returns a list of dictionaries containing article titles, links, and content.
    """
    driver.get("https://www.tpointtech.com/")
    wait_and_get(By.XPATH, "//input[contains(@id, 'searchInput')]").send_keys(context + Keys.RETURN)

    results, scraped_urls = [], set() 
    total_results = 0
    pages_scraped = 0

    # Wait for modal results to load
    wait_and_get(By.ID, "modalResults")

    # Get total results from the info element
    info_element = wait_and_get(By.CSS_SELECTOR, "#modalResults > div.text-right > small") # This element contains the total results info
    if info_element:
        match = re.search(r'Showing\s+\d+\s*–\s*\d+\s+of\s+(\d+)', info_element.text) # Regex to find total results
        total_results = int(match.group(1)) if match else 0 

    pagination_buttons = get_pagination_buttons()
    total_pages = len(pagination_buttons)
    print(f"Total pages detected: {total_pages}")

    # If no pagination buttons found, assume only one page
    for page_num in range(total_pages):
        pagination_buttons = get_pagination_buttons()
        if page_num >= len(pagination_buttons):
            break

        try:
            # Click the pagination button
            driver.execute_script("arguments[0].click();", pagination_buttons[page_num])
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#modalResults a")))
            pages_scraped += 1 # Increment the page count
            print(f"Scraping page {pages_scraped}...")

            # Get all links on the current page
            current_links = []
            links = wait_and_get(By.CSS_SELECTOR, "div#modalResults a", multiple=True)
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and href not in scraped_urls:
                        current_links.append({"title": link.text.strip(), "link": href}) # Store title and link
                        scraped_urls.add(href)
                except StaleElementReferenceException:
                    continue

            if not current_links:
                print("No new links found on this page.")
                continue
            
            # Iterate through each link and extract content
            for item in current_links:
                # Open each link in a new tab to extract content
                original_window = driver.current_window_handle
                try:
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.get(item["link"])
                    content = extract_content()
                    results.append({
                        "Title": item["title"],
                        "Link": item["link"],
                        "Content": content
                    })
                except Exception as e:
                    print(f"Error loading {item['link']}: {e}")
                finally:
                    driver.close()
                    driver.switch_to.window(original_window)
            # Check if we have reached the total results limit
            if total_results and len(scraped_urls) >= total_results:
                break

        except Exception as e:
            print(f"Failed to scrape page {page_num + 1}: {e}")
            continue

    print(f"Finished scraping {len(scraped_urls)} unique articles.")
    return results

def save_chunked(results, out_path):
    """ 
    Saves the scraped results into a JSONL file with chunked content.
    Each article's content is split into chunks of 500 characters.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    chunked_data = []
    for article in results:
        # Chunk the content of each article
        chunks = textwrap.wrap(' '.join(article["Content"]), width=500) # Chunk by 500 characters
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "title": article["Title"],
                "chunk_id": idx + 1,
                "content": chunk,
                "source": "tpointtech.com"
            })
    with open(out_path, "a", encoding="utf-8") as f:
        for item in chunked_data:
            json.dump(item, f)
            f.write("\n")

# Run
query = "machine learning interview" #data analyst, SQL
scraped_results = scrape_details(query)
save_chunked(scraped_results, "data/data.jsonl")

driver.quit()
