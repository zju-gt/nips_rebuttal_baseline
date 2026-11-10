from openai import OpenAI

# 初始化客户端
# base_url 必须指向你服务器的地址，并带上 /v1
# api_key 在 vLLM 中默认不需要，但 SDK 要求必须传一个字符串，写 "EMPTY" 即可
client = OpenAI(
    api_key='SET_VIA_LLM_API_KEY',
    base_url="https://routify.alibaba-inc.com/protocol/openai/v1",
)

# 调用聊天接口
completion = client.chat.completions.create(
    model="gpt-5.2-2025-12-11",  # 这里的 model 必须和你启动服务时的 --served-model-name 一致
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    max_tokens=1024,
    temperature=0.7
)

# 打印回复
print("模型回答：")
print(completion.choices[0].message.content)
