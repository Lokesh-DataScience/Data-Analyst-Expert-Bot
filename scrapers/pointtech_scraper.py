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

# Setup WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
wait = WebDriverWait(driver, 10)

def wait_and_get(by, value, multiple=False):
    try:
        if multiple:
            return wait.until(EC.presence_of_all_elements_located((by, value)))
        else:
            return wait.until(EC.presence_of_element_located((by, value)))
    except TimeoutException:
        return [] if multiple else None

def extract_content():
    content = []
    try:
        container = driver.find_element(By.TAG_NAME, "td")
        for tag in ['p', 'ol', 'ul']:
            for element in container.find_elements(By.TAG_NAME, tag):
                for sub in element.find_elements(By.TAG_NAME, "li") if tag in ['ol', 'ul'] else [element]:
                    text = sub.text.strip()
                    if text:
                        content.append(text)
    except NoSuchElementException:
        pass
    return content

def get_pagination_buttons():
    try:
        return driver.find_elements(By.CSS_SELECTOR, "span.pagination-number")
    except Exception:
        return []

def scrape_details(context: str):
    driver.get("https://www.tpointtech.com/")
    wait_and_get(By.XPATH, "//input[contains(@id, 'searchInput')]").send_keys(context + Keys.RETURN)

    results, scraped_urls = [], set()
    total_results = 0
    pages_scraped = 0

    # Wait for modal results to load
    wait_and_get(By.ID, "modalResults")

    # Try to get total count
    info_element = wait_and_get(By.CSS_SELECTOR, "#modalResults > div.text-right > small")
    if info_element:
        match = re.search(r'Showing\s+\d+\s*–\s*\d+\s+of\s+(\d+)', info_element.text)
        total_results = int(match.group(1)) if match else 0

    pagination_buttons = get_pagination_buttons()
    total_pages = len(pagination_buttons)
    print(f"Total pages detected: {total_pages}")

    for page_num in range(total_pages):
        pagination_buttons = get_pagination_buttons()
        if page_num >= len(pagination_buttons):
            break

        try:
            driver.execute_script("arguments[0].click();", pagination_buttons[page_num])
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div#modalResults a")))
            pages_scraped += 1
            print(f"Scraping page {pages_scraped}...")

            current_links = []
            links = wait_and_get(By.CSS_SELECTOR, "div#modalResults a", multiple=True)
            for link in links:
                try:
                    href = link.get_attribute("href")
                    if href and href not in scraped_urls:
                        current_links.append({"title": link.text.strip(), "link": href})
                        scraped_urls.add(href)
                except StaleElementReferenceException:
                    continue

            if not current_links:
                print("No new links found on this page.")
                continue

            for item in current_links:
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

            if total_results and len(scraped_urls) >= total_results:
                break

        except Exception as e:
            print(f"Failed to scrape page {page_num + 1}: {e}")
            continue

    print(f"Finished scraping {len(scraped_urls)} unique articles.")
    return results

def save_chunked(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    chunked_data = []
    for article in results:
        chunks = textwrap.wrap(' '.join(article["Content"]), width=500)
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
query = "powerbi" #data analyst, SQL
scraped_results = scrape_details(query)
save_chunked(scraped_results, "data/data.jsonl")

driver.quit()
