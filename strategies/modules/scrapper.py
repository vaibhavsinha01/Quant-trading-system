import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from utils.resource_path import resource_path


def scrap():
    # Set up Chrome options
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.90 Safari/537.36")

    # Start the WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    # Navigate to NSE historical index data page
    driver.get("https://www.nseindia.com/reports-indices-historical-index-data")
    time.sleep(5)  # Wait for page JavaScript

    try:
        logging.info("Attempting to find and select index dropdown...")
        index_select = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select[class='form-control no-border-radius']"))
        )
        logging.info("Dropdown found, selecting NIFTY 50...")
        Select(index_select).select_by_visible_text("NIFTY 50")
        time.sleep(2)

        logging.info("Attempting to click 1Y button...")
        one_year_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "ul.dayslisting li a[data-val='1Y']"))
        )
        driver.execute_script("arguments[0].click();", one_year_button)
        time.sleep(7)  # Wait for data to load

        logging.info("Waiting for table data to be populated...")
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#historic-repot-table-reference tbody tr td"))
        )

        logging.info("Extracting table data...")
        table = driver.find_element(By.ID, "historic-repot-table-reference")
        rows = table.find_elements(By.TAG_NAME, "tr")

        headers = [th.text.strip() for th in rows[0].find_elements(By.TAG_NAME, "th")]
        logging.info(f"Found headers: {headers}")

        data = []
        for row in rows[1:]:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) == len(headers):
                data.append([col.text.strip() for col in cols])

        logging.info(f"Number of rows extracted: {len(data)}")
        logging.info(f"Number of columns: {len(headers)}")

        if not data:
            raise Exception("No data was extracted from the table")

        df = pd.DataFrame(data, columns=headers)
        logging.info("DataFrame created successfully")
        logging.info(f"DataFrame shape: {df.shape}")
        logging.info(df.head())

        df.to_csv(resource_path("storage/data/nifty50.csv"), index=False)
        logging.info("Data saved to nifty50_historical_data.csv")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

        with open("page_source.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)
        logging.info("Page source saved to page_source.html")

        driver.save_screenshot("error_screenshot.png")
        logging.info("Screenshot saved as error_screenshot.png")

    finally:
        driver.quit()
