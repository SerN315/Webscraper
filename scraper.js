const { Builder, Browser, By, Key, until } = require("selenium-webdriver");
const createCsvWriter = require("csv-writer").createObjectCsvWriter;

(async function scrape() {
  let driver = await new Builder().forBrowser(Browser.EDGE).build();
  try {
    await driver.get("https://phimmoiyyy.net/");
    var names = await driver.findElements(By.css(".owl-item"));
    const data = [];
    for (n of names) {
      data.push({ title: await n.getText() });
    }
    const csvWriter = createCsvWriter({
      path: "output.csv",
      header: [{ id: "title", title: "Title" }],
    });
    await csvWriter.writeRecords(data);
    console.log("Data saved to output.csv");
  } finally {
    await driver.quit();
  }
})();
