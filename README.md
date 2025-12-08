## course tech review project
### Installment
```
conda create -n raptor -y python=3.9
conda activate raptor
git clone https://github.com/hexuan21/lite_raptor.git
cd lite_raptor
pip install -r requirements.txt
```

### Run demo
```
export OPENAI_BASE_URL=""
export OPENAI_API_KEY=""
```

```
python eval/eval_bm25.py --model_name "gpt-4o-mini"
python eval/eval_dpr.py --model_name "gpt-4o-mini"

python eval/eval_vanilla_llm.py --context_type "ten_psg" --model_name "gpt-4o-mini"
python eval/eval_vanilla_llm.py --context_type "no_context" --model_name "gpt-4o-mini"
python eval/eval_vanilla_llm.py --context_type "all_corpus" --model_name "gpt-4o-mini"

python eval/eval_raptor.py --model_name "gpt-4o-mini"
```
