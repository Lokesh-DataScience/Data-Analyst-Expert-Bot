from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')
import textwrap
import json

# Automatically download and use ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

def scrape_details(context: str):
    driver.get("https://www.geeksforgeeks.org/")
    time.sleep(2)  # Wait for page to load

    # Locate the search bar and enter the context
    search_bar = driver.find_element(By.XPATH, "//input[contains(@class, 'HomePageSearchContainer_homePageSearchContainer_container_input__1LS0r')]")
    search_bar.send_keys(context)
    search_bar.send_keys(Keys.RETURN)
    
    time.sleep(3)  # Wait for results to load

    # Get all article elements
    results = []

    while True:  
        # Extract all articles on the current page
        articles = driver.find_elements(By.CLASS_NAME, "ResultArticle_articleContainer__84glf")
        articles_info = []
        for article in articles:
            try:
                title_element = article.find_element(By.CLASS_NAME, "ResultArticle_articleContainer__headerLink___pap7")
                title = title_element.text
                link_element = article.find_element(By.TAG_NAME, "a")
                link = link_element.get_attribute("href")
                articles_info.append({"title": title, "link": link})
            except:
                continue

        # results = []
        for info in articles_info:
            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(info["link"])
            time.sleep(2)
            try:
                paragraphs = driver.find_elements(By.XPATH, '//div[@class="text"]//p')
                content = "\n".join([p.text for p in paragraphs if p.text.strip()])
            except Exception as e:
                print(f"Failed to extract content from: {info['link']}")
                content = f"Error: {str(e)}"

            results.append({
                "Title": info["title"],
                "Content": content
            })
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            time.sleep(1)
         # Try to click the Next button
        try:
            # Look for all pagination buttons
            next_buttons = driver.find_elements(By.CLASS_NAME, "PaginationContainer_paginationContainer__link__qTC3z")
            clicked = False

            for btn in next_buttons:
                if "Next" in btn.text and btn.is_enabled():
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(3)
                    clicked = True
                    break

            if not clicked:
                print("No 'Next' button found. Reached last page.")
                break  # Exit the loop if there's no Next button

        except Exception as e:
            print(f"Pagination ended or failed: {e}")
            break


    return results

title = "ms excel for data analysis" #data analysis, data analysis tutorials, powerbi course, sql for data analysis
results = scrape_details(title)

chunked_data = []
for article in results:
    chunks = textwrap.wrap(article["Content"], width=500)  # chunk by 500 chars
    for idx, chunk in enumerate(chunks):
        chunked_data.append({
            "title": article["Title"],
            "chunk_id": idx + 1,
            "content": chunk,
            "source": "geeksforgeeks.org"
        })

with open("data/gfg_data.jsonl", "a", encoding="utf-8") as f:
    for item in chunked_data:
        json.dump(item, f)
        f.write("\n")


driver.quit()