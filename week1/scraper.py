from bs4 import BeautifulSoup
import requests

headers = {
    "User_Agent": "Mozilla/5.0 (Windows NT 10.0; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_website_contents(url: str) -> str:
    """
    Return the title & contents of the website at the given url
    """
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip = True)
        return title + "\n\n" + text
    except Exception as e:
        print(f"Error fetching website contents: {e}")
        return None

'''
1️⃣ Make an HTTP GET request
response = requests.get(url, headers=headers)


Sends an HTTP GET request to the given url

headers usually contain things like:

    User-Agent (to look like a browser)

    Authorization tokens (if needed)

The result is stored in response, which contains:

    response.status_code → HTTP status (200, 404, etc.)

    response.content → raw HTML bytes

    response.text → decoded text (string)

2️⃣ Parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")


Converts the raw HTML into a BeautifulSoup object

"html.parser" is Python’s built-in HTML parser

soup now represents the webpage as a tree structure (DOM)

This allows you to search, filter, and modify HTML elements easily.

3️⃣ Extract the page title
title = soup.title.string if soup.title else "No title found"


soup.title:

    Finds the <title> tag in the HTML

.string:

    Extracts the text inside <title>...</title>

Conditional check:

    If no <title> exists, it avoids an error and uses "No title found"

✅ Example:

<title>OpenAI Documentation</title>


Result:

title = "OpenAI Documentation"

4️⃣ Remove irrelevant elements
for irrelevant in soup.body(["script", "style", "img", "input"]):


Finds all tags inside <body> matching:

<script> → JavaScript code

<style> → CSS

<img> → images

<input> → form fields

These elements are not useful for text extraction

soup.body([...]) is shorthand for:

soup.body.find_all(["script", "style", "img", "input"])

5️⃣ Delete those elements from the DOM
irrelevant.decompose()


Completely removes the tag and its contents from the HTML tree

After this, those elements no longer exist in soup

This prevents junk text like JavaScript or CSS from appearing in the output.

6️⃣ Extract visible text
text = soup.body.get_text(seperator="\n", strip = True)


⚠️ Small typo here:
seperator ❌ → should be separator ✅

Correct version:

text = soup.body.get_text(separator="\n", strip=True)


What this does:

Extracts all visible text from <body>

separator="\n" → puts each text block on a new line

strip=True → removes leading/trailing whitespace

Result:

Heading
Paragraph text
Another paragraph

7️⃣ Combine title and body text
return title + "\n\n" + text


Combines:

Page title

Two newlines for readability

Cleaned body text

Returns a single string ready for:

LLM input

Search indexing

Summarization

Vector embedding
'''