import streamlit as st
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
# import matplotlib.pyplot as plt
# サンプルデータ準備、あらかじめ用意されている。
from sklearn.datasets import load_iris
#非線形変換
import torch.nn.functional as F
import re

st.title('あやめの分類')
st.caption('ディープラーニング学習の成果としてPyTorchチュートリアルにクイックスタートのサンプルのデータを「あやめ」に変え学習させたモデルの動作確認サイト')
st.markdown('###### 詳細は')
link = '[イチゲブログ](https://kikuichige.com/17482/)'
st.markdown(link, unsafe_allow_html=True)

warning=None
iris = load_iris()
# 入力値と目標値を抽出
x=iris['data']
t=iris['target']
ori_x=x
ori_t=t
ayame_list=[]
for index, word in enumerate(ori_x):
    ayame_list.append(f'{index}　特徴量{word}　種類{ori_t[index]}')

# modelを定義します
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4,6),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(6,3)#あやめ3種類の判別なのでノード
         )
    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# 訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
device = "cuda" if torch.cuda.is_available() else "cpu"
st.text(f'デバイスは{device}です')

#modelの読み込み
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))

with st.form(key='profile_form'):
#セレクトボックス
    st.text(f'種類　0:setosa 1:versicolor 2:virginica')
    testid=st.selectbox(
        'あやめの特徴量のデータセットを選んで表示を押してください。（全部で150個あります）',
       (ayame_list)
        )
    ayame_list_input='mada'
    test_input=st.text_input('任意の特徴量でもテストできます！特徴量を4つ,区切りで入力してください。上のセレクトボックスで入力する場合、ここは空欄にしてください。')
    submit_btn=st.form_submit_button('表示')

    if test_input:
        ayame_list_input=test_input.split(',')
        if len(ayame_list_input)!=4:
            warning='4つの数字をカンマ区切りで入力してください'
        else:
            ayame_list_raw= [float(i) for i in ayame_list_input]
            ayame_list_test = torch.tensor(ayame_list_raw, dtype = torch.float32)

    if submit_btn:
        pass
    classes = [
        "0：setosa",
        "1：versicolor",
        "2：virginica",
        "入力した特徴量と近いものをセレクトボックスで確認してみてください"
    ]

    model.eval()

    # 正規表現で先頭の数字を抽出
    match = re.match(r'\d+', testid)
    # match.group()で抽出した文字列を取得
    first_num =match.group()
    if test_input and not warning:
        x=ayame_list_test
        y=torch.tensor(3,dtype=torch.int64)
    else:
        x=torch.tensor(ori_x[int(first_num)],dtype=torch.float32)
        y=torch.tensor(ori_t[int(first_num)],dtype=torch.int64)

    with torch.no_grad():
        pred = model(x)
        
        predicted, actual = classes[pred.argmax(0)], classes[y]
        if submit_btn:
            if not warning:
                 st.write(f'種類予測: "{predicted}", 種類正解: "{actual}"')
            else:
                st.write(warning)
        warning = None

