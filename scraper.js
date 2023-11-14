const { Builder, Browser, By, Key, until } = require("selenium-webdriver");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;

// Xác định hàm scraping
(async function scrape() {
  // Khởi tạo trình duyệt và driver
  let driver = await new Builder().forBrowser(Browser.EDGE).build();
  try {
    // Truy cập vào trang web
    await driver.get("https://phimmoiyyy.net/");

    // Tìm phần tử link "2023" và click vào nó
    let linkElement = await driver.findElement(By.linkText("2023"));
    await linkElement.click();

    // Chờ cho đến khi URL chứa "/nam-phat-hanh/2023"
    await driver.wait(until.urlContains("/nam-phat-hanh/2023"), 5000);

    // Lấy danh sách các phần tử article trên trang
    let articleElements = await driver.findElements(By.css("article.item"));

    const data = [];

    // Lặp qua 3 phần tử article đầu tiên
    for (let i = 0; i < articleElements.length; i++) {
      let articleElement = articleElements[i];

      // Lấy phần tử link tiêu đề và lấy thuộc tính href
      let titleLinkElement = await articleElement.findElement(By.css("h3 a"));
      let link = await titleLinkElement.getAttribute("href");

      // Truy cập vào trang con
      await driver.get(link);

      // Lấy các thông tin cần thiết từ trang con
      let imgElement = await driver.findElement(By.css(".poster img"));
      let titleElement = await driver.findElement(By.css("h1"));
      let spanContentElement = await driver.findElement(
        By.css(".extra .valor")
      );
      let dateElement = await driver.findElement(By.css(".date"));
      // ...

      // Xử lý trường hợp không tìm thấy phần tử tapElement
      let tapElement;
      try {
        tapElement = await driver.findElement(
          By.css(".movie_label .item-label")
        );
      } catch (error) {
        console.log("Tap element not found");
        tapElement = null;
      }

      // Lấy các thông tin khác từ trang con
      let ratingValueElement = await driver.findElement(
        By.css(".starstruck-rating span.dt_rating_vgs")
      );
      let ratingCountElement = await driver.findElement(
        By.css(".starstruck-rating span.rating-count")
      );
      let genresElements = await driver.findElements(By.css(".sgeneros a"));

      // Lấy giá trị của thuộc tính src từ phần tử img
      let imgUrl = await imgElement.getAttribute("src");

      // Lấy nội dung của các phần tử khác
      let title = await titleElement.getText();
      let spanContent = await spanContentElement.getText();
      let dateCreated = await dateElement.getText();
      let tap = "";
      if (tapElement) {
        tap = await tapElement.getText();
      }
      let ratingValue = await ratingValueElement.getText();
      let ratingCount = await ratingCountElement.getText();

      // Lấy các thể loại từ các phần tử a trong phần tử có class "sgeneros"
      let genres = [];
      for (let j = 0; j < genresElements.length; j++) {
        let genreElement = genresElements[j];
        let genre = await genreElement.getText();
        genres.push(genre);
      }

      // Tạo đối tượng dữ liệu cho hàng dữ liệu hiện tại
      let rowData = {
        imgUrl,
        title,
        link,
        spanContent,
        dateCreated,
        tap,
        ratingValue,
        ratingCount,
        genres,
      };

      // Thêm hàng dữ liệu vào mảng data
      data.push(rowData);

      // Quay lại trang danh sách
      await driver.get("https://phimmoiyyy.net/");
      linkElement = await driver.findElement(By.linkText("2023"));
      await linkElement.click();
      await driver.wait(until.urlContains("/nam-phat-hanh/2023"), 5000);
      articleElements = await driver.findElements(By.css("article.item"));
    }

    // Tạo đối tượng csvWriter để ghi dữ liệu vào file CSV
    const csvWriter = createCsvWriter({
      path: "output.csv",
      header: [
        { id: "imgUrl", title: "URL Hình ảnh" },
        { id: "title", title: "Tiêu đề" },
        { id: "link", title: "Link" },
        { id: "spanContent", title: "Nội dung" },
        { id: "dateCreated", title: "Ngày phát hành" },
        { id: "tap", title: "Tập" },
        { id: "ratingValue", title: "Đánh giá" },
        { id: "ratingCount", title: "Số lượt đánh giá" },
        { id: "genres", title: "Thể loại" },
      ],
    });

    // Ghi dữ liệu vào file CSV
    await csvWriter.writeRecords(data);
  } finally {
    // Đóng trình duyệt và driver
    await driver.quit();
  }
})();
