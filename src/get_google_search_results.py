# import googlesearch
from googleapiclient.discovery import build

import settings

my_api_key = settings.API_KEY
my_cse_id = settings.CSE_ID

def google_search(query, api_key=my_api_key, cse_id=my_cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']


# def get_search_results_urls(query):
#     return googlesearch.search(query=query, tld='com', lang='en', start=0, stop=10)


def get_google_search_results(search_term: str, num_results: int = 10):
    # search_term = 'Lung cancer treatment' + search_term

    # result = get_search_results_urls(search_term)
    # for i in result:
    #     print(i)
    results = google_search(search_term, num=num_results)
    links = [result['link'] for result in results]
    titles = [result['title'] for result in results]
    snippets = [result['snippet'] for result in results]
    return links, titles, snippets

