import xml
import re
from urllib.parse import urlparse

import scrapy
from scrapy.exporters import CsvItemExporter
from scrapy.shell import inspect_response
from scrapy.utils.response import open_in_browser


class BigLungCancer(scrapy.Spider):
    name = 'nslc_treatment'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [kwargs.get('start_url')]

    def parse(self, response):
        self.logger.info("Processing:" + response.url)

        text = [' '.join(line.strip() for line in p.xpath('.//text()').extract() if line.strip()) for p in response.xpath('//p')]
        base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(response.url))
        # paragraphs = response.xpath('//p')
        # largest_p = paragraphs[0]
        # for p in paragraphs:
        #     if len(''.join(p.xpath('.//text()').getall())) > len(''.join(largest_p.xpath('.//text()').getall())):
        #         largest_p = p

        yield {'base_url': base_url,
               'text': text,
               }
