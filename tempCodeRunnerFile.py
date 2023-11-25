from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium import webdriver
from csv import DictWriter
import time

def scrape():
    options = Options()
    options.headless = False  # Set to True if you want it to run in the background

    driver = webdriver.Edge()

    try:
        # Truy cập vào trang web
        
        driver.get("https://phimmoiyyy.net/")
        time.sleep(2)

        # Tìm phần tử link "2023" và click vào nó
        link_element = driver.find_element(By.LINK_TEXT, "2023")
        link_element.click()
        time.sleep(2)


        while True:
            
            # Chờ cho đến khi URL chứa "/nam-phat-hanh/2023"
            WebDriverWait(driver, 5).until(EC.url_contains("/nam-phat-hanh/2023"))
            current_page_url = driver.current_url  # Store the current page UR
            # Lấy danh sách các phần tử article trên trang
            article_elements = driver.find_elements(By.CSS_SELECTOR, "article.item")

            data = []

            # Lặp qua các phần tử article
            for i in range(len(article_elements)):
                article_element = article_elements[i]

                # Lấy phần tử link tiêu đề và lấy thuộc tính href
                title_link_element = article_element.find_element(By.CSS_SELECTOR, "h3 a")
                link = title_link_element.get_attribute("href")

                # Truy cập vào trang con
                driver.get(link)

                # Lấy các thông tin cần thiết từ trang con
                info_element = driver.find_element(By.CSS_SELECTOR, "#info p")
                title_element = driver.find_element(By.CSS_SELECTOR, "h1")
                span_content_element = driver.find_element(By.CSS_SELECTOR, ".extra .valor")
                date_element = driver.find_element(By.CSS_SELECTOR, ".date")
                rating_value_element = driver.find_element(By.CSS_SELECTOR, ".starstruck-rating span.dt_rating_vgs")
                rating_count_element = driver.find_element(By.CSS_SELECTOR, ".starstruck-rating span.rating-count")
                genres_elements = driver.find_elements(By.CSS_SELECTOR, ".sgeneros a")

                # Lấy giá trị của thuộc tính src từ phần tử info
                

                # Lấy nội dung của các phần tử khác
                info = info_element.text
                title = title_element.text
                span_content = span_content_element.text
                date_created = date_element.text
                rating_value = rating_value_element.text
                rating_count = rating_count_element.text

                # Lấy các thể loại từ các phần tử a trong phần tử có class "sgeneros"
                genres = [genre_element.text for genre_element in genres_elements]

                # Tạo đối tượng dữ liệu cho hàng dữ liệu hiện tại
                row_data = {
                    "info": info,
                    "title": title,
                    "spanContent": span_content,
                    "dateCreated": date_created,
                    "ratingValue": rating_value,
                    "ratingCount": rating_count,
                    "genres": genres
                }

                # Thêm hàng dữ liệu vào mảng data
                data.append(row_data)

                # Quay lại trang danh sách
                driver.get(current_page_url)
                article_elements = driver.find_elements(By.CSS_SELECTOR, "article.item")

            # Ghi dữ liệu vào file CSV
            # Ghi dữ liệu vào file CSV
            with open("output.csv", "a", newline="", encoding="utf-8") as csv_file:
             fieldnames = [
        "title", "info", "link", "spanContent",
        "dateCreated", "ratingValue", "ratingCount", "genres"
    ]
             writer = DictWriter(csv_file, fieldnames=fieldnames)

            # Check if the file is empty (no headers), then write headers
            if csv_file.tell() == 0:
             writer.writeheader()
             writer.writerows(data)

         # Check if there's a next page
            next_page_links = driver.find_elements(By.CSS_SELECTOR, '.pagination a #nextpagination')
            next_page_link = next((link for link in next_page_links if 'inactive' not in link.get_attribute('class')), None)

            if next_page_link:
             next_page_number = int(next_page_link.text) if next_page_link.text.isdigit() else 0

            if not next_page_link or current_page_url.endswith(f"page/{next_page_number}"):
               break  # Exit loop if there's no active next page or if it's on the same page

    # Click the next page link using JavaScript
            driver.execute_script("arguments[0].click();", next_page_link)
            time.sleep(2)  # Add a slight delay for the new page to load

    # Wait for the new page to load completely
            WebDriverWait(driver, 10).until(EC.url_changes(current_page_url))
            current_page_url = driver.current_url  # Update the current page URL for the next iteration



    finally:
        # Close the browser
        driver.quit()

# Call the scrape function
scrape()
