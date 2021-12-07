import urllib
from urllib.request import urlopen  #the library for url acesss
import re #regular expressions
from bs4 import BeautifulSoup
import mechanicalsoup        #basically is a headless browser that works with python
front_page_pdb = 'https://www.rcsb.org/'  #advanced search page

#open page
#the main html parse method i three steps
page = urlopen(front_page_pdb)
html_bytes = page.read()
html = html_bytes.decode("utf-8") #this is what you get from the incpect element browser of your function

#scrapping html using exact string matching
title_index = html.find("<title>")
#print(title_index)  #this is now the index of the start of the title tag
start_index = title_index + len("<title>")
end_index = html.find("</title>")
title = html[start_index:end_index]

#this method is less reliable then regular expressions, for example people might render title as <title > with a space or class, rendering this usless
#but is useful for us since were scrapping the same page constantly here. will user regex tho if we have to scrap into the PDB entry descriptions, which can be less uniform

#regex scraping of html 

findall_test = re.findall("[Ss]earch",html)  #returns a list(?)
search_test = re.search("[Ss]earch",html)     # returs a MatchObject

#print(findall_test[0:3])
#print(search_test)
#print(search_test.group())  #retruns the first most inclusive result in some way; see other ways to interact with MatchObject

pattern = "<title.*?>.*?</title.*?>"
match_results = re.search(pattern, html, re.IGNORECASE)

search = match_results.group()  #carefull this results in errors if object ends up empty

#using a specific html parsing thingy like beautiful soup - a screen scrapping library

soup = BeautifulSoup(html, "html.parser")  #html parser s built into python by default
#print(soup.get_text())  #strangely this results in an unsupported browser message?
#print(soup.get_text)     
#find all img tags, using findall on the results of the soup object
images = soup.find_all("img")
#print(images)
#print(images[3].name)  #retuns type of tag
#print(images[1]["src"])#source of image


#soup doesent help much with forms; which is actually what we want from pdb!
#using the mechanical soup browser

browser = mechanicalsoup.Browser()
page = browser.get(front_page_pdb)
print(page)  #page is a Response type object! 200 is the sucess ok state 

# note mechanical soup uses beautifoul soup to parse html and is infacte an upgrade on top of it: page has a soup attribute
#print(page.soup)

#using a form

search_html = page.soup
#forms = search_html.select('div[id="search-bar-component"]')  #doesent work
#forms = search_html.select('form')

#form = forms[0]

#form["value"] = "1N5U"

#next_page = browser.submit(form, page.url)
