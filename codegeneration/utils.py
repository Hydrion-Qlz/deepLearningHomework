import requests

def send_notice(content):
    token = "b16d758c011643308409b67f4447d4dd"
    title = "训练结束"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=html"
    response = requests.request("GET", url)
    print("消息发送成功", response.text)

# send_notice(f"训练正确率:55%\n测试正确率:96.5%")
