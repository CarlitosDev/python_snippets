'''
test_playwright.py

Playwright is intended to superseed Chromium


pip3 install pytest-playwright
playwright install

Install docs: 
https://playwright.dev/python/docs/intro


'''


from playwright.sync_api import sync_playwright
playwright = sync_playwright().start()

url_chatgpt = 'http://chat.openai.com/'

# Use playwright.chromium, playwright.firefox or playwright.webkit
# Pass headless=False to launch() to see the browser UI
# browser = playwright.chromium.launch()
# browser = playwright.chromium.launch(headless=False)
# it does not pass the 'not human' verification
browser = playwright.chromium.launch_persistent_context(headless=False)
page = browser.new_page()
page.goto(url_chatgpt)


# page.screenshot(path="example.png")
browser.close()
playwright.stop()