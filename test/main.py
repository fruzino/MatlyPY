import logging
from pyplot import matmodel

logging.basicConfig(level=logging.WARNING)

knowledge = """
hello there human
i am a robot built with matpy
i like to help you code python
the weather is digital in my world
coding is fun and robots are helpful
"""

print("Chatbot")
print("Training the brain... (please wait)")

_, weights, vocab = matmodel.model(
    type=1, 
    data=knowledge, 
    epochs=10000, 
    learning_rate=0.1
)

print("Brain ready! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ").lower()
    
    if user_input == "exit":

        matmodel.save("chatbot_brain.gguf", weights, vocab)
        print("Brain saved to chatbot_brain.gguf. Goodbye!")
        break

    response, _, _ = matmodel.model(
        type=1,
        instruct=user_input,
        weights=weights,
        vocab=vocab,
        token=8,
        temp=0.5
    )

    print(f"Robot: {response}\n")