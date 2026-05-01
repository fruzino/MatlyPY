"""
The following code is given by AI. 
This is a test script. Human made scripts are not in the
AI folder.
"""
from matlypy import model

def search_and_study(query, num_tokens=20):
    print(f"Searching for: {query}...")
    search_url = "https://en.wikipedia.org/wiki/N-gram" 

    print(f"Fetching content from {search_url}...")
    clean_text = model.tools.fetch(search_url)

    if "Error" in clean_text:
        return "Failed to retrieve data."

    print("Training model on web data...")
    output, brain, vocab = model.model(
        data=clean_text, 
        instruct=query.split()[0],
        token=num_tokens, 
        n=3
    )

    return output

result = search_and_study("natural language processing")
print("\nModel Output based on Web Search:")
print(result)