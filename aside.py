import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt



def wordcloud():
    EVAL_DATA_PATH="eval_data/hotpotqa_100.json"
    data=json.load(open(EVAL_DATA_PATH,"r"))
    all_contexts=[]
    for item in data:
        context_str=""
        for x in item["context"]:
            context_str+="\n"+x[0]+"\n"+"".join(x[1])
        all_contexts.append(context_str)
    
    all_context_one_str="\n\n".join(all_contexts).lower()
    useless_words=[" the ", " a ", " an "," to "," with ","caption","content"]
    for w in useless_words:
        all_context_one_str.replace(w," ")
    
    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200
    )
    wc.generate(all_context_one_str)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    wc.to_file("assets/hotpotqa_wordcloud.png")


def demo():
    import os
    from raptor import RetrievalAugmentation

    # Initialize with default configuration. For advanced configurations, check the documentation. [WIP]
    RA = RetrievalAugmentation()

    question = "How did Cinderella reach her happy ending?"

    SAVE_PATH = "eval_res/hotpotqa_tree_2025-12-05--04:52:38"

    RA = RetrievalAugmentation(tree=SAVE_PATH)
    ret_res=RA.retrieve(question=question)
    print(ret_res)
    print(RA.retriever.tree.num_layers)
    answer = RA.answer_question(question=question)


if __name__ =="__main__":
    # wordcloud()
    demo()