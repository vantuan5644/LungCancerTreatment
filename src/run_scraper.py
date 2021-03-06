import os
import re
from urllib.parse import urlparse

import numpy as np
import scrapy
from scrapy import signals
from scrapy.crawler import CrawlerProcess
from scrapy.exporters import CsvItemExporter

import settings
from src.get_google_search_results import get_google_search_results


class BigLungCancer(scrapy.Spider):
    name = 'nslc_treatment'

    def __init__(self, category='', **kwargs):
        super().__init__(**kwargs)  # python3

    def parse(self, response):
        stages = getattr(self, 'stages', None)
        base_url = '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(response.url))
        base_urls = ['{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(url)) for url in self.start_urls]

        texts = []

        """
        First pattern:
        <h>
        <p>
        <p>
        """

        for first_node, following_node in [('h2', 'p'), ('h3', 'p')]:
            for stage in stages:
                stage = stage.lower()
                self.logger.debug(stage)
                selectors = response.xpath(
                    f'//*[preceding-sibling::{first_node}[contains(concat(" ", translate(., "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), " ")," {stage} ")]]')
                for selector in selectors:
                    tag = selector.xpath('name()').get()
                    if tag == following_node:
                        paragraph = ' '.join(
                            line.strip() for line in selector.xpath('.//text()').extract() if line.strip())
                        if len(paragraph) > 0:
                            # paragraph = re.sub('\.,', '\. ', paragraph)
                            texts.append(paragraph)
                    else:
                        break
        """
        Second pattern: 
        <h>
        <div>
          <p>
          <p>
        """
        if len(texts) == 0:
            for first_node, following_node in [('h2', 'p'), ('h3', 'p')]:
                for stage in stages:
                    selector = response.xpath(
                        f'//{first_node}[contains(concat(" ", translate(., "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), " "), " {stage} ")]/parent::node()//{following_node}//node()')
                    paragraph = ' '.join(
                        line.strip() for line in selector.xpath('.//text()').extract() if line.strip())
                    if len(paragraph) > 0:
                        texts.append(paragraph)

        titles = getattr(self, 'titles', None)
        snippets = getattr(self, 'snippets', None)
        if base_url in base_urls:
            index = base_urls.index(base_url)
            title = titles[index]
            snippet = snippets[index]
        else:
            title = None
            snippet = None

        self.logger.debug(texts)
        yield {'url': response.url,
               'title': title,
               'snippet': snippet,
               'text': texts,
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
    # stages = map(lambda x: f'"stage {x}"', range(5))
    """
    There are some "hyperparameters" (some kind of hard-coding things) that can affect the results:
    1. The keyword (search term) to get input URLs from Google, i.e: stage 0
    2. The keyword again, but in other forms, i.e: stage 3A or stage IIIA
    3. CSS tag to get the content:
      - Type 1: <ul> <li>
      - Type 2: <h2> <p>
    """
    stages_map = {
        'stage 0': ['stage 0'],
        'stage 1': ['stage 1', 'stage I'],
        'stage 2': ['stage 2', 'stage II'],
        'stage 3a': ['stage 3a', 'stage IIIA'],
        'stage 3b': ['stage 3b', 'stage IIIB'],
        'stage 4': ['stage 4', 'stage IV'],
    }
    # The other forms of the keyword

    for i, (keyword, stages) in enumerate(stages_map.items()):
        csv_file = os.path.join(settings.PROJECT_ROOT, 'data_crawled', f'{keyword}.csv')
        if os.path.exists(csv_file):
            os.remove(csv_file)

        final_urls, final_titles, final_snippets = [[]] * 3

        search_terms = [f'non small lung cancer treatment {np.random.choice(stages)}',
                        'non small lung cancer treatment by stage'
                        ]
        for search_term in search_terms:
            urls, titles, snippets = get_google_search_results(search_term=search_term)
            final_urls = list(set(final_urls + urls))
            final_titles = list(set(final_titles + titles))
            final_snippets = list(set(final_snippets + snippets))
        process = CrawlerProcess(settings={'FEED_FORMAT': 'csv',
                                           'FEED_URI': csv_file,
                                           'ROBOTSTXT_OBEY': True})

        process.crawl(BigLungCancer, start_urls=final_urls, titles=final_titles, snippets=final_snippets, stages=stages)
    process.start()  # The script will block here until the crawling is finished
