import time

import pandas as pd
import os
from urllib.parse import urlparse

import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.exporters import CsvItemExporter

from src.get_google_search_results import lung_cancer_treatment_result


class BigLungCancer(scrapy.Spider):

    name = 'nslc_treatment'

    def __init__ (self, category='', **kwargs):
        super().__init__(**kwargs)  # python3

    def parse(self, response):
        self.logger.info("Processing:" + response.url)
        text = [' '.join(line.strip() for line in p.xpath('.//text()').extract() if line.strip()) for p in
                response.xpath('//p')]
        base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(response.url))
        base_urls = ['{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(url)) for url in self.start_urls]
        # paragraphs = response.xpath('//p')
        # largest_p = paragraphs[0]
        # for p in paragraphs:
        #     if len(''.join(p.xpath('.//text()').getall())) > len(''.join(largest_p.xpath('.//text()').getall())):
        #         largest_p = p
        index = base_urls.index(base_url)
        title = self.titles[index]
        snippet = self.snippets[index]

        yield {'url': response.url,
               'title': title,
               'snippet': snippet,
               'text': text,
               }


class LungCancerPipeline(object):
    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls()
        crawler.signals.connect(pipeline.spider_opened, signals.spider_opened)
        crawler.signals.connect(pipeline.spider_closed, signals.spider_closed)
        return pipeline

    def spider_opened(self, spider):
        self.file = open('output.csv', 'w+b')
        self.exporter = CsvItemExporter(self.file)
        self.exporter.start_exporting()

    def spider_closed(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        self.exporter.export_item(item)
        return item


if __name__ == "__main__":
    stages = map(lambda x: f'"stage {x}"', range(5))
    for i, keyword in enumerate(stages):
        print(keyword)
        csv_file = f'data_crawled/stage_{i}.csv'
        if os.path.exists(csv_file):
            os.remove(csv_file)
        start_urls, titles, snippets = lung_cancer_treatment_result(keyword, num_results=10)
        process = CrawlerProcess(settings={'FEED_FORMAT': 'csv',
                                           'FEED_URI': csv_file,
                                           'ROBOTSTXT_OBEY': False})

        process.crawl(BigLungCancer, start_urls=start_urls, titles=titles, snippets=snippets)
    process.start()  # the script will block here until the crawling is finished
