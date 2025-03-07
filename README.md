목표: Vision Transformer 모델을 사전학습 없이, Cifar10 데이터셋에 대해 Validation accuracy 90% 달성

Vision Transformer(ViT)에 대한 간단한 직관:
1. Transformer가 문장에서 단어를 각각의 토큰으로 취급하여, Self-attention 메커니즘을 통해 context에 따른 단어 사이의 관계를 파악함
2. Vision Transformer는 이미지를 패치 단위로 쪼개고, 단어가 아니라 패치를 토큰으로 입력하여 Self-attention을 수행함
3. CNN은 Convolution kernel이 이미지를 local receptive field의 방식으로 지역적인 정보들을 포착하는데 유리함
4. Transformer는 이미지 전체 패치에 대해 Self-attention을 수행하기 때문에 이미지 패치 사이의 global context를 포착하기 유리함
5. 한계는 입력 토큰 길이의 제곱에 연산량이 비례함 (O(N^2))

구성 요소:
![image](https://github.com/user-attachments/assets/ac1dac91-b488-44fd-9250-11e77e09b286)
1. patch_embed.py: 입력 이미지를 패치 단위로 쪼개고, linear projection하여 패치 임베딩을 만듦 -> class token과 positional encoding을 더해줌
2. msa.py: Multi-head Self-attention을 수행함, 이미지 패치 사이의 context를 파악함
3. transformer_encoder.py: Multi-head Self-attention과 Skip connection, Layer normalization, MLP를 포함 -> 입력과 출력 차원이 같음
4. mlp.py: Transformer encoder를 지나서 마지막에 10개 class probability를 예측함
5. vit.py
6. train.py

실험: visdom 라이브러리로 시각화
![image](https://github.com/user-attachments/assets/6ae9f0b3-5cff-40a0-93a6-fe854d2632c6)
1. Transformer encoder 1개: validation accuracy 66.8%
2. Transformer encoder 4개: validation accuracy 81.6%

