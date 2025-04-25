import requests
from bs4 import BeautifulSoup

def search_indian_kanoon(query):
    search_query = query.replace(" ", "+")
    url = f"https://indiankanoon.org/search/?formInput={search_query}+doctypes:supremecourt,highcourts"

    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    results = []

    for result_block in soup.select(".result"):
        
        a_tag = result_block.find("a")
        title = a_tag.get_text(strip=True)
        url = "https://indiankanoon.org" + a_tag["href"]


        headline_div = result_block.find_next("div", class_="headline")
        snippet = headline_div.get_text(strip=True) if headline_div else None

        hlbottom = result_block.find_next("div", class_="hlbottom")
        court = hlbottom.select_one(".docsource").get_text(strip=True) if hlbottom else None

        results.append({
            "title": title.strip().replace("  ", " "),
            "url": url.strip().replace("  ", " "),
            "snippet": snippet.strip().replace("  ", " "),
            "court": court.strip().replace("  ", " ")
        })

    return results

def main():
    query = "section 376"
    result = search_indian_kanoon(query)
    print(result)
    return result

if __name__ == "__main__":
    main()