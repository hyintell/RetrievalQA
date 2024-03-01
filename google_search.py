import serpapi
from utils import load_file, save_file_jsonl

serpapi_api_key = "<Your SerApi API key>"

def call_search_engine(query):

    params = {
        "q": query,
        "engine": "google", # Set parameter to google to use the Google API engine
        # "location": "California, United States",
        "hl": "en", # Parameter defines the language to use for the Google search.
        "gl": "au", # Parameter defines the country to use for the Google search.
        "google_domain": "google.com",  # Parameter defines the Google domain to use.
        "api_key": serpapi_api_key,

    }

    results = serpapi.search(**params)

    return results


def parse_google_research_results(results, retrieved_num=5):
    
    retrieved_docs = []
    # Answer box has higher priority
    if 'answer_box' in results:
        parsed_item = {}
        answer_box = results["answer_box"]
        if "title" in answer_box:
            parsed_item["title"] = answer_box["title"]
        if "snippet" in answer_box:
            parsed_item["text"] = answer_box["snippet"]
    
        print(f"answwer_box: {parsed_item}")
        retrieved_docs.append(parsed_item)

    if "organic_results" in results:
        items = results['organic_results']
        print(f"# organic_results: {len(items)}")
        if len(items) < retrieved_num:
            retrieved_num = len(items)

        for idx in list(range(len(items)))[:retrieved_num]:
            item = items[idx]
            parsed_item = {}
            if "title" in item:
                parsed_item["title"] = item['title']
            # if "snippet_highlighted_words" in item: 
            #     highlights = item['snippet_highlighted_words']
            if "snippet" in item:
                parsed_item["text"] = item['snippet']
            # if "link" in item:
            #     link = item['link']
                
            retrieved_docs.append(parsed_item)
        
    return retrieved_docs



def main():

    input_file_path = "./data/retrievalqa.jsonl"

    # load input data
    input_data = load_file(input_file_path)
    print(f"input data: {input_file_path}, #: {len(input_data)}")
    print(input_data[0])

    # only using Google search for questions from realtimeqa and freshqa
    data_sources = ["realtimeqa", "freshqa"]

    query_count = 0
    for item in input_data:
        if item["data_source"] not in data_sources:
            continue 
        
        query_count += 1
        query = item["question"]

        results = call_search_engine(query)
        retrieved_docs = parse_google_research_results(results, retrieved_num=5)

        item["context"] = retrieved_docs

        print(f"query: {query}")
        print(f"source: {item['data_source']}, # context: {len(item['context'])}")
        print(item["context"])

    print(f"total query times: {query_count}")

    # sanity check
    count_of_empty_context = sum([1 if len(item["context"]) == 0 else 0 for item in input_data])
    assert count_of_empty_context == 0

    save_file_jsonl(input_data, "./data/retrieved_docs.jsonl")
    


if __name__ == "__main__":
    main()

