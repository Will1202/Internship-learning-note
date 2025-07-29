*这个聊天机器人还不会使用文档检索功能——只是进行基本的问答*
# 步骤 1：安装所需的库
在开始编码之前，我们需要安装必要的库。
(Google colab)
```
!pip install llama-index-llms-gemini llama-index
```

# 第 2 步：设置 API 密钥
要使用Gemini，我们需要设置一个 API 密钥。

**为什么？**

您之所以设置 API 密钥，是因为它就像一个密码，让您的聊天机器人能够与Gemini Pro对话。没有它，您的聊天机器人将无法向 Gemini 提问或获得回复。API 密钥会告知 Google您有权使用他们的 AI 服务。
想象一下获取 Wi-Fi 密码——没有它，您就无法连接到互联网！
 
**获取免费的 Gemini API 密钥**
- 前往Google AI Studio：https://aistudio.google.com/
- 使用您的 Google 帐户登录。
- 导航到API 密钥并生成新密钥。
- 复制您的 API 密钥。

**步骤 4：运行您的聊天机器人**
- 在Google Colab或本地运行该脚本。
- 向聊天机器人询问任何问题并看看它如何回答！
- 尝试用不同的模型进行实验，例如"**gemini-pro**"。

