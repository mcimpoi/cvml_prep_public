from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def get_seed_urls(fname: str) -> list:
    with open(fname, "r") as f:
        lines = f.read().splitlines()

    return {line.split(",")[0]: line.split(",")[1] for line in lines}


if __name__ == "__main__":
    options = Options()
    options.add_argument("--headless=new")

    urls_dict = get_seed_urls("data/dtd_extra/queries/unsplash.txt")

    for query, url in urls_dict.items():
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        figures = driver.find_elements(by=By.XPATH, value="//figure[@itemprop='image']")
        print(len(figures))
        _ = figures[-1].location_once_scrolled_into_view
        figures = driver.find_elements(by=By.XPATH, value="//figure[@itemprop='image']")
        print("After scroll: ", len(figures))
        print(figures[0].find_elements(by=By.TAG_NAME, value="a"))
        break
