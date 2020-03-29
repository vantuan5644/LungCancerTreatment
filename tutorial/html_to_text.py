import json
from html.parser import HTMLParser


class HTMLFilter(HTMLParser):
    text = ""

    def handle_data(self, data):
        self.text += data


if __name__ == "__main__":
    f = HTMLFilter()
    data = json.load('big_lung_cancer.json')['main-content']

    f.feed(data)
    print(f.text)
