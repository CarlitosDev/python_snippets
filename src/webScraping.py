Web scraping
------------

Selenium vs Beautiful soup

Why might you consider using Selenium? 
- Selenium is first of all a tool writing automated tests for web applications. 
- Pretty much entirely to handle the case where the content you 
want to crawl is being added to the page via JavaScript, 
rather than baked into the HTML. 


Tricks with Selenium:

driver   = webdriver.Firefox();
urlToCheck = 'http://www.theperfumeshop.com/p/1220422'
driver.get(urlToCheck)

input_element = driver.find_element_by_class_name('description')

driver.find_element_by_css_selector('div[class=''description''] > dl>dt ').text