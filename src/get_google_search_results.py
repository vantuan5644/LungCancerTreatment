import googlesearch
from googleapiclient.discovery import build

from src import settings

my_api_key = settings.API_KEY
my_cse_id = settings.CSE_ID


def google_search(query, api_key=my_api_key, cse_id=my_cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key, cache_discovery=False)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']


def get_search_results_urls(query):
    return googlesearch.search(query=query, tld='com', lang='en', start=0, stop=10)


def lung_cancer_treatment_result(stage: str, num_results: int):
    search_term = 'Lung cancer treatment' + stage
    # result = get_search_results_urls(search_term)
    # for i in result:
    #     print(i)
    results = google_search(search_term, num=num_results)
    results = [result['link'] for result in results]
    return results


if __name__ == "__main__":
    lung_cancer_treatment_result(stage='"stage 2"', num_results=10)
