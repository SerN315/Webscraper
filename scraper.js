const { Builder, Browser, By, Key, until } = require("selenium-webdriver");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;

// Xác định hàm scraping
(async function scrape() {
  // Tạo một instance của WebDriver (trong trường hợp này, sử dụng Microsoft Edge)
  let driver = await new Builder().forBrowser(Browser.EDGE).build();
  try {
    // Điều hướng đến trang web mục tiêu
    await driver.get("https://phimmoiyyy.net/");
    let linkElement = await driver.findElement(By.linkText("2023"));
    // Nhấp vào liên kết để chuyển hướng sang trang con
    await linkElement.click();
    // Đợi cho trang con được tải hoàn thành
    await driver.wait(until.urlContains("/nam-phat-hanh/2023"), 5000);
    // Tìm tất cả các phần tử div với class "owl-item"
    let divElements = await driver.findElements(By.css("article.item"));

    // Tạo một mảng để lưu trữ dữ liệu cho mỗi div dưới dạng một object
    const data = [];

    // Lặp qua các phần tử div
    for (let i = 0; i < divElements.length; i++) {
      let divElement = divElements[i];

      // Tìm các phần tử con bên trong phần tử div
      let imgElement = await divElement.findElement(By.css("img"));

      let titleLinkElement;
      try {
        titleLinkElement = await divElement.findElement(By.css("h3 a"));
      } catch (error) {
        console.log("Không thể tìm thấy phần tử tiêu đề:", error.message);
        continue; // Bỏ qua lần lặp này nếu không tìm thấy phần tử tiêu đề
      }

      let spanElement;
      try {
        spanElement = await divElement.findElement(By.css("span")); // CSS selector ví dụ
      } catch (error) {
        console.log("Không thể tìm thấy phần tử span:", error.message);
        continue; // Bỏ qua lần lặp này nếu không tìm thấy phần tử span
      }

      // Lấy nội dung văn bản và thuộc tính của các phần tử con
      let imgUrl = await imgElement.getAttribute("src");
      let title = await titleLinkElement.getText();
      let link = await titleLinkElement.getAttribute("href");

      // Thực thi mã JavaScript để lấy nội dung văn bản của phần tử span bị tràn
      let spanContent = await driver.executeScript(
        "return arguments[0].innerText;",
        spanElement
      );

      // Lưu trữ dữ liệu trong một object
      let rowData = {
        imgUrl,
        title,
        link,
        spanContent,
        trangthai: await divElement.findElement(By.css(".trangthai")).getText(),
      };

      // Đẩy object vào mảng data
      data.push(rowData);
    }

    // Tạo một instance của CSV writer với các tiêu đề dựa trên các khóa của object đầu tiên trong mảng data
    const csvWriter = createCsvWriter({
      path: "output.csv",
      header: [
        { id: "imgUrl", title: "URL Hình ảnh" },
        { id: "title", title: "Tiêu đề" },
        { id: "link", title: "Liên kết" },
        { id: "spanContent", title: "Nội dung span" },
        { id: "trangthai", title: "Trạng thái" },
      ],
    });

    // Ghi dữ liệu vào tệp CSV
    await csvWriter.writeRecords(data);
    console.log("Dữ liệu đã được lưu vào output.csv");
  } finally {
    // Kết thúc instance của WebDriver và đóng trình duyệt
    await driver.quit();
  }
})();
