import os
import json
import requests

MAX_TRY=3

def _llm_azure_api(
    user_prompt,
    model_name,
    chat_config={},
):
    from openai import AzureOpenAI
    endpoint = os.environ["AZURE_ENDPOINT_URL"]

    subscription_key = os.environ["AZURE_OPENAI_API_KEY"]
    api_version = "2024-12-01-preview"
    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            max_tokens=4096,
            temperature=1.0,
            top_p=1.0,
            model=model_name
        )
        res=str(response.choices[0].message.content)
        return res
    
    except Exception as e:
        print(f"[ERR] error {e}")
        return ""


def _llm_openrouter_api(
    user_prompt,
    model_name="gpt-4o-mini",
    chat_config={},
    ):
    api_key=os.environ["OPENAI_API_KEY"]
    llm_base_url=os.environ["OPENAI_BASE_URL"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"{llm_base_url}/chat/completions"

    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                ]
            }
        ]
        
    payload = {
        "model": model_name,
        "messages": messages, 
        "max_tokens": chat_config.get("max_tokens", 1024),
        "temperature": chat_config.get("temperature", 1.0)
    }
    thinking_enabled=chat_config.get("thinking_enabled",False)
    if thinking_enabled == True:
        payload["reasoning"]={
            "exclude": False,
            "max_tokens": chat_config.get("thinking_budget", 2048)
        }
        payload["max_tokens"] = chat_config.get("max_tokens", 1024) + chat_config.get("thinking_budget", 2048)
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print(f"[INFO] Open Router response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERR] Open Router API error: {response.text}")
            return ""
        
        thinking = str(response.json()['choices'][0]['message'].get('reasoning', ''))
        output = str(response.json()['choices'][0]['message'].get('content', ''))
        
        if thinking_enabled:
            res = "<think>"+thinking+"</think>"+"\n"+output
        else:
            res = output
        print("currently model's answer:", res)
        return res
    except Exception as e:
        print(f"[ERR] error {e}")
        return ""