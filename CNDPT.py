from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.edge.options import Options
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from csv import DictWriter
import time
import os
from selenium.webdriver.common.action_chains import ActionChains

def scrape():
    options = Options()
    options.add_argument('--headless')
    driver = webdriver.Chrome()
    try:
        # Truy cập vào trang web
        driver.get("https://www.lazada.vn/")
        search_element = driver.find_element(By.XPATH, "//input[@class='search-box__input--O34g']")
        search_element.send_keys("smartphone") 
        search_button = driver.find_element(By.XPATH, "//button[@class='search-box__button--1oH7']")
        search_button.click()
        current_page_url = driver.current_url  # Lưu trữ URL trang hiện tại
        products_elements = driver.find_elements(By.CSS_SELECTOR, ".Bm3ON")
        data=[]
        star_urls=[]
        star_urls_per = []
        comment_content =""
        d_name_spans=""
        for i in range(min(len(products_elements),10)):
            products_element = products_elements[i]
            title_link_element = products_element.find_element(By.CSS_SELECTOR,".Ms6aG .qmXQo .ICdUp ._95X4G a")
            link = title_link_element.get_attribute("href")
            driver.get(link)
            time.sleep(2)
            title_element= driver.find_element(By.XPATH,"//h1[@class='pdp-mod-product-badge-title']")
            body = driver.find_element(By.TAG_NAME,'body')
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(20)
            summary_element = driver.find_element(By.CSS_SELECTOR,".mod-rating .content .left .summary")
            rating_score = summary_element.find_element(By.CLASS_NAME, "score")
            rating_score_average = rating_score.find_element(By.CLASS_NAME, "score-average")
            star_score_average = summary_element.find_element(By.CSS_SELECTOR, ".average .container-star")
            star_images = star_score_average.find_elements(By.TAG_NAME,"img")
            for star_image in star_images:
                star_url = star_image.get_attribute("src")
                star_urls.append(star_url)
            rating_count = summary_element.find_element(By.CSS_SELECTOR, ".count")
            filter_button = driver.find_element(By.CSS_SELECTOR, ".pdp-mod-filterSort .oper[data-spm-anchor-id='a2o4n.pdp_revamp.ratings_reviews.i0.604a41cacAcTDZ']")
            filter_button.click()
            time.sleep(20)
            li_elements = driver.find_element(By.CSS_SELECTOR,".next-menu-content li")
            column_mapping = {
                0:"All-Stars",
                1:"5 Stars",
                2:"4 Stars",
                3:"3 Stars",
                4:"2 Stars",
                5:"1 Stars",
            }
            for index in range(1,len(li_elements)):
                li_element = li_elements[index]
                if li_element.get_attribute("class") != "next-menu-item disabled":
        # Click on the <li> element
                   li_element.click()
                   time.sleep(2)  # Wait for the comments to update
                comments = driver.find_elements(By.CSS_SELECTOR, ".mod-reviews .item")
                comment_contents = []  # List to store comment_content for all items
                for comment in comments:
                    star_score_per = comment.find_element(By.CSS_SELECTOR, ".top .container-star")
                    star_images_per = star_score_per.find_elements(By.TAG_NAME, "img")
                    star_urls_per = []
                    for star_image_per in star_images_per:
                        star_url_per = star_image_per.get_attribute("src")
                        star_urls_per.append(star_url_per)
                    d_name_spans = comment.find_elements(By.CSS_SELECTOR, ".middle span")
                    comment_content = comment.find_element(By.CSS_SELECTOR, ".item-content .content")
                    comment_text = comment_content.text
                    comment_contents.append(comment_text)  # Append comment_text to the list


            # Lấy nội dung của các phần tử khác
            title = title_element.text
            score = rating_score_average.text
            numbs = rating_count.text 
            username = d_name_spans[0].text
            column_value = column_mapping[index]
            # verify = d_name_spans[2].text

            row_data={
                "TenSP":title,
                "DG":score,
                "DG_average_image":star_urls,
                "SoDG":numbs,
                "TenHienthi":username,
                "DG_rieng":column_value,
                # "comment_DG":star_urls_per,
                "comment":comment_contents

            }

            data.append(row_data)

            driver.back()
            products_elements = driver.find_elements(By.CSS_SELECTOR, ".Bm3ON")

        # Kiểm tra xem tập tin đã tồn tại chưa để xác định việc ghi headers hay không
        write_headers = not os.path.exists("lazada.csv")

        # Ghi dữ liệu vào file CSV
        with open("lazada.csv", "w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                    "TenSP", "DG", "SoDG","username","DG_rieng","comment"
                ]
            writer = DictWriter(csv_file, fieldnames=fieldnames)
            write_headers = True

            if write_headers:
                writer.writeheader()  # Ghi headers chỉ khi tập tin mới

            writer.writerows(data)

    finally:
        # Đóng trình duyệt
        driver.quit()

# Call the function
scrape()
