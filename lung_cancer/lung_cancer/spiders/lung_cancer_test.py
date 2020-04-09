import xml
import re
from urllib.parse import urlparse

import scrapy
from scrapy.shell import inspect_response
from scrapy.utils.response import open_in_browser


class NonSmallLungCancer(scrapy.Spider):
    name = 'ns_lung_cancer'

    start_urls = ['https://www.cancer.org/cancer/lung-cancer/treating-non-small-cell/by-stage.html',
                  # 'https://www.lungcancer.org/find_information/publications/163-lung_cancer_101/269-non_small_cell_lung_cancer_treatment',
                  # 'https://www.cancer.net/cancer-types/lung-cancer-non-small-cell/types-treatment',
                  ]

    def parse(self, response):
        self.logger.info("Processing:" + response.url)

        base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(response.url))

        if base_url == 'https://www.cancer.org/':
            title = response.xpath("//div/h1/text()").extract_first()
            main_content = response.xpath("//div[contains(@class, 'text-ckeditor')]//following-sibling::node()").getall()
        elif base_url == 'https://www.lungcancer.org/':

            title = response.xpath("//div/h2/text()").extract_first()
            main_content = response.xpath("//div[contains(@class, 'content')]//node()").getall()
        else:

            title = response.xpath("//div/h2/text()").extract_first()
            main_content = response.xpath("//div[contains(@class, 'field-item')]//node()").getall()

        # yield {'title': title}
        # ref_text = response.xpath("//p/a[contains(@href, '')]").xpath('normalize-space(.)').extract()

        for node in main_content:
            if node[0] == '<':
                node_type = node[0:node.find('>')+1]
                yield {node_type: self.remove_tags(node)}

    @staticmethod
    def remove_tags(text):
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text).replace('\n', '').strip()

    def parse_details(self, response, item=None):
        if item:
            # populate more `item` fields
            return item
        else:
            self.logger.warning('No item received for %s', response.url)
            inspect_response(response, self)

        if "item name" not in response.body:
            open_in_browser(response)
