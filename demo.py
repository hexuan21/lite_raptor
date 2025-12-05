import os
from raptor import RetrievalAugmentation

# Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
RA = RetrievalAugmentation()

with open('eval_data/sample.txt', 'r') as file:
    text = file.read()
RA.add_documents(text)

question = "How did Cinderella reach her happy ending?"
answer = RA.answer_question(question=question)
print("Answer: ", answer)

SAVE_PATH = "demo/cinderella"
os.makedirs(os.path.dirname(SAVE_PATH),exist_ok=True)
RA.save(SAVE_PATH)

RA = RetrievalAugmentation(tree=SAVE_PATH)
answer = RA.answer_question(question=question)