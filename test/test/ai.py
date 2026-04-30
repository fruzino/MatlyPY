from matlypy import model

corpus = """

"""

response, brain, vocab = model.model(
    data=corpus,
    instruct="Hello", 
    token=10,        
    n=2, 
    temp=0.7        
)

print(f"Bot: {response}")