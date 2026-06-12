from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import warnings
warnings.filterwarnings('ignore')
import textwrap
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# Automatically download and use ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

def scrape_details(context: str):
    """
    Scrapes articles from Towards Data Science based on the provided context.
    """
    driver.get("https://towardsdatascience.com/")
    time.sleep(2)  # Wait for page to load

    search_button = driver.find_element(By.CLASS_NAME, "wp-block-tenup-search-button__button") # Click on the search button
    search_button.click()
    time.sleep(1)  # Wait for search bar to appear
    search_bar = driver.find_element(By.XPATH, "//input[@class='wp-block-search__input']")
    search_bar.send_keys(context) # Type the search context
    search_bar.send_keys(Keys.RETURN) # Press Enter to search

    time.sleep(3)  # Wait for search results to load

    results = []
    page_count = 0
    max_pages = 10  # Limit to 10 pages

    # Loop through the pages of search results
    while page_count < max_pages:
        articles = driver.find_elements(By.CLASS_NAME, "wp-block-post")
        articles_info = []
        # Extract article titles and URLs
        for article in articles:
            try:
                article_link = article.find_element(By.XPATH, ".//h2[contains(@class, 'has-link-color')]//a")
                title = article_link.text.strip()
                url = article_link.get_attribute("href")
                articles_info.append({"title": title, "url": url})
            except Exception as e:
                print(f"Error extracting article info: {e}")
                continue
        # Iterate through the articles and extract content
        for info in articles_info:
            # Open a new tab for each article to extract content
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(info["url"])
            time.sleep(2)
            try:
                # Extract the content of the article
                paragraphs = driver.find_elements(By.CSS_SELECTOR, "div.entry-content.wp-block-post-content")
                content = "\n".join([p.text for p in paragraphs if p.text.strip()])
            except Exception as e:
                print(f"Failed to extract content from: {info['url']}")
                content = f"Error: {str(e)}"
            results.append({
                "Title": info["title"],
                "URL": info["url"],
                "Content": content
            })
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            time.sleep(1)
        page_count += 1 # Increment page count
        try:
            # Wait for pagination nav
            pagination_nav = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "nav[aria-label='Pagination']"))
            )

            # Wait for next button to be clickable
            next_button = WebDriverWait(pagination_nav, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a.wp-block-query-pagination-next"))
            )

            # Scroll into view
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", next_button)
            driver.execute_script("arguments[0].click();", next_button)

        except TimeoutException:
            print("Pagination or next button not clickable in time.")
            break

        except Exception as e:
            print(f"Error navigating to next page: {e}")
            break

    return results

if __name__ == "__main__":
    context = "machine learning interview"
    results = scrape_details(context)
    chunked_data = []
    for article in results:
        chunks = textwrap.wrap(article["Content"], width=500)  # chunk by 500 chars
        for idx, chunk in enumerate(chunks):
            chunked_data.append({
                "title": article["Title"],
                "chunk_id": idx + 1,
                "content": chunk,
                "source": "towardsdatascience.com"
            })

    with open("data/data.jsonl", "a", encoding="utf-8") as f:
        for item in chunked_data:
            json.dump(item, f)
            f.write("\n")


    driver.quit()