import matlypy
from matlypy import model

text = """The cat is fake! But why? I donot know whay?F"""

while True:
    prompt = input("> ")

    output, brain, vocab = model.model(
        data=text,
        instruct=prompt,
        token=640,
        n=20,
        temp=2
    )

    print(output)