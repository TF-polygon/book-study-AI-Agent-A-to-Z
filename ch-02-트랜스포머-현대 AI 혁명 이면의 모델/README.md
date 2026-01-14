# 2장: 트랜스포머: 현대 AI 혁명 이면의 모델

핵심 주제: 1장에서 살펴본 모델 (RNN, LSTM, GRU, CNN)의 한계점을 극복하기 위한 어텐션 메커니즘 및 트랜스포머 개념 학습

- 어텐션과 셀프 어텐션 탐구하기
- 트랜스포머 모델 소개
- 트랜스포머 학습하기
- 마스크드 언어 모델링 탐구하기
- ~~내부 메커니즘 시각화하기~~
- ~~트랜스포머 활용하기~~

## 1. 어텐션과 셀프 어텐션 탐구하기
### 1.1. 기술적 배경
#### 기계 번역 개념의 근본적인 실패 이유
- 기계 번역 문제가 겉으로 보이는 것보다 실제로 훨씬 복잡한 과제
- 당시 (1950~60년대) 컴퓨터의 연산 능력의 부족
- 학습에 필요한 데이터의 부족
1990년대 이후, 인터넷이 보급되면서 방대한 양의 데이터를 확보할 수 있게 되었고, `GPU`가 등장하면서 충분한 연산 능력도 갖추게 되었다. <br>

#### 기계 번역 과제에 적용된 기존 모델
`RNN (Recurrent Neural Network) 기반 seq2seq 모델`이 가장 널리 사용된 시스템
```
# seq2seq
[RNN Module:Encoder] → [RNN Module:Decoder]
```
`Encoder` : 번역하고자 하는 입력 문장으로부터 중요한 정보를 보존. [Context Vector 생성 모듈]<br>
`Decoder` : 입력 받은 Context vector로부터 실제 번역 문장을 생성한다. [번역 문장 생성 모듈]<br>

#### RNN 기반 기계 번역 모델의 문제점
1. 정렬 문제
- Input, Output의 길이가 서로 다르기 때문에 발생하는 성능 문제
- `RNN Module:Encoder`는 고정된 크기의 컨텍스트 벡터에 문장의 의미를 압축해서 담아야 하는 구조적 한계
2. 기울기 소실 및 폭발 문제
- 역전파 과정에서 학습 신호가 여러 단계를 거쳐 점점 사라지거나 반대로 강해져 학습이 제대로 안되지 않는 현상 발생
- `기울기 소실`: 앞쪽 부분이 제대로 학습되지 않아 긴 문맥을 파악할 수 없음
- `기울기 폭발`: 학습 과정이 불안정해져서 모델이 제대로 작동하지 않음
3. 병렬 처리 불가능과 메모리 한계
- 순차 처리 방식으로 인한 비효율적인 학습, 병렬 처리 사용 불가능

### 1.2. Attention Mechanism
`Attention Mechanism`: 입력 텍스트에 존재하는 단어들과 번역된 텍스트의 단어들 간의 관계를 학습하는 메커니즘 (정렬 문제 해결)
<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 1.</b> 정렬 문제의 예시.
</p>

<b>핵심 아이디어</b>: 입력 시퀀스에서 중요한 부분만 집중하여 핵심적인 문맥 정보를 전달 <br>
- 모든 단어를 차별없이 보는 대신, 지금 번역하고 있는 단어와 직접적으로 연관된 입력 단어에만 가중하는 방식 <br>

Decoding 과정에서 각 토큰은 다른 언어에서 대응하는 <b>특정 정보를 탐색</b> 및 그 시점에서 <b>어떤 입력 토큰이 중요한지를 결정</b><br>

#### Attention Mechanism의 1단계
`Encoder`의 은닉 상태 $h$와 `Decoder`의 이전 출력 상태 $s$를 정렬

$$e_{i, j}=\rm{score}(s_{i-1}, h_j)$$

이 단계에서 얻는 정보는 $h$와, $s$ 사이의 유사도를 나타내는 스칼라 값 $e_{i, j}$ 1개

#### Attention Mechanism의 2단계
$e_{i, j}$를 `softmax` 함수를 통해 [0, 1] 범위로 변환

이 값이 클수록 입력 단어와 번역 중인 단어의 관련도가 높다는 의미 <br>

$$a_{i, j} = \rm{softmax}(e_{i, j}) = \frac{exp(e_{i, j})}{\sum_{k=1}^t exp(e_{i, k})}$$
- $a_{i, j}$: attention score
- $exp()$: 자연 상수 $e$ 를 밑으로 하는 지수 함수

이를 통한 번역 과정은 단순히 순서 나열이 아니라, 현재 번역되는 단어가 전체 중 어떤 부분과 연결되는지를 확률적으로 판단하는 과정 <br>

#### Attention Mechanism의 3단계
 각 `Encoder` 은닉 상태 $h_j$와 attention score $a_{i, j}$를 곱한 뒤 모두 더해 가중합 계산

$$c_t=\sum_{j=1}^T a_{i,j} \cdot h_j$$
- $c_t$: 가중합

이를 통해 전체 은닉 상태 집합 의 정보를 담고 있는 고정 길이 `Context vector` 즉, 요약된 핵심 맥락을 얻는다. 
- 번역 과정에서, `Context vector`는 계속 업데이트되며, 입력 시퀀스의 각 부분에 얼마만큼 주의를 기울여야 하는지를 동적으로 알려주는 것
- 번역 중, `Decoder`가 다음 단어를 번역할 때마다 `Encoder`에서 매 시점마다 새롭게 계산되어 `Decoder`로 전달
- 모델은 원문의 어느 부분을 참고하는지 잊지 않고 유지 가능
<br>

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 2.</b> Attention mechanism을 적용한 모델이 계산한 원문 단어와 목표단어 간의 가중합 히트맵.
</p>

### 1.3. Attention Mechanism의 장점
- 어텐션은 초기 상태로 가는 직접 연결 경로를 제공하여 기울기 소실 문제를 완화
- `Encoder`가 번역 과정에서 원문에 직접 접근할 수 있어 병목 현상을 줄임
- 어떤 단어가 정렬에 사용되는지 확인할 수 있어 해석 가능성 향상
- '해석 가능성'을 확보
- 특정 단어 매칭 이유를 시각적으로 확인이 가능하기 때문에, 블랙박스가 아닌 설명 가능한 인공지능 (XAI; explainable AI)의 성격을 띔

### 1.4. Self Attention
핵심: <b> 입력 자체에서 직접 정보를 추출 </b> <br>
- 번역처럼 `Input → Output`이라는 구조가 아니더라도, 하나의 문장 안에서 단어들 사이의 관계를 파악하는 데 바로 활용할 수 있는 메커니즘

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 3.</b> Self Attention Mechanism
</p>

- 하나의 입력 문장 내부에서 단어들끼리 서로 어떤 관계를 맺고 있는지를 파악하는데 집중

$$\rm{Attention}(Q,K,V)=\rm{softmax} (\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$$
- $Q$: Query
- $K$: Key
- $V$: Value
- $d$: 시퀀스의 크기

$Attention(O,K,V)$는 $Q$와 $K$를 내적을 통해 비교하여 중요도를 계산하고, `softmax` 함수로 상대적인 중요도를 계산하여 [0, 1] 범위로 값을 정규화, 그리고 $V$를 그 중요도에 따라 가중합하는 과정을 압축적으로 표현한 수식 <br>

<!-- softmax 적용 수식  -->

벡터의 길이가 길어질수록 내적 값이 지나치게 커지고, `softmax` 함수의 입력 값이 너무 커져 확률 질량이 일부 요소에만 쏠리게 된다. 그 결과, 기울기가 매우 작아지는 문제가 발생한다.

<b>Solution</b> : `Scaled dot-product attention` → 내적 결과를 차원 수 $D$의 제곱근으로 나누어 정규화
- 현대 LLM의 주요 아키텍처인 `Transformer` 모델에서도 기본 단위로 사용

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 4.</b> 전개된 Self Attention
</p>

<b>핵심적인 차이</b>: 세 개의 가중치 행렬 (Query $Q$, Key $K$, Value $V$)사용

- 처음에는 각 행렬 무작위 초기화
- $Q$는 현재 주목하는 대상을 나타냄
- $K$는 모델에 이전 입력에 대한 정보 제공
- $V$는 최종 입력 정보를 추출하는 역할

`Self Attention`의 첫 번째 단계: 입력 $X$ (각 토큰을 벡터로 표현한 배열)에 이 세 행렬을 곱함

$$ Q = X \cdot W^Q,~K=X \cdot W^K,~V=X \cdot W^V $$
<b>장점</b>
- 동일한 입력으로부터 여러 표현을 동시에 추출할 수 있다는 점
- 입력에 대해 다양한 관점의 관계를 학습할 수 있다
- 연산 병렬화가 가능하여, `Multi-head Attention` 구현 가능
- 같은 입력 문장을 넣더라도 Q, K, V라는 세 가지 서로 다른 관점으로 투영하여 단어간 관계를 다각도로 바라보게 된다.

### 1.5. Multi-head Self Attention
<b>핵심</b>: 모델이 입력 시퀀스 안에 존재하는 다양한 관계를 동시에 포착 <br>
한 문장의 특정 단어가 문맥적으로 여러 단어와 동시에 연결될 수 있기 때문에 매우 중요한 부분 <br>
- 학습 과정에서 각 헤드의 K와 Q 행렬은 서로 다른 유형의 관계를 모델링하도록 특화된다.

```
The big cat sat on the mat.
Head 1: (문법 관계) sat → cat
Head 2: (수식 관계) big → cat
Head 3: (위치 관계) on, the mat → cat
```
<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 5.</b> Multi-head Self Attention
</p>

`Multi-head` 구조로 인해 모델은 '단어 A와 단어 B의 관계', '단어 A와 단어 C의 관계'처럼 서로 다른 연관성을 동시에 학습할 수 있게 됨

<b>장점</b>
- 각 입력에 대해 다양한 표현 추출 가능
- 모든 연산 병렬 처리 가능. 각 Head는 독립적으로 연산 가능
- 반드시 `Encoder-Decoder` 구조의 모델이 아니더라도 활용 가능
- `RNN`처럼 여러 시점을 거치지 않아도 멀리 떨어진 단어 쌍 간의 관계를 즉시 파악 가능

<b>단점</b>
- 토큰 수 $N$에 따라 연산 비용이 제곱으로 증가하며 순서에 대한 고유한 개념이 없음
- 긴 문서를 다룰수록 연산량과 메모리 사용량이 빠르게 늘어 계싼 비용이 많이 듦

<b>계산 비용 및 공간 복잡도 2차 함수</b>
$$\rm{time} = \it{O(T^2+d)}$$
$$\rm{space} = \it{O(T^2 +Td)}$$
- $T$: 시퀀스 길이
- $d$: 각 벡터의 차원

## 2. 트랜스포머 모델 소개

### 2.1. Self Attention 등장 이후에도 존재하는 기계 번역의 과제
- 모델이 문장의 의미를 제대로 포착하지 못하고 여전히 오류가 많다
- 모델이 학습 때 본적 없는 단어들을 처리하기 어렵다
- 대명사 및 기타 문법적 형태를 다룰 때 오류가 발생한다
- 긴 텍스트에서 문맥을 유지하기 어렵다
- 학습 데이터셋과 테스트 데이터셋의 도메인이 다를 경우 일반화 능력이 떨어져서 성능이 저하된다.
- RNN은 구조적 한계로 병렬 연산이 불가능해 반드시 순차적으로 계산해야 한다. (연산 비효율성)

2016년 구글에서 RNN을 개선하는 대신 아예 제거하는 발상으로, `Transformer`를 제안한 논문 'Attention Is All You Need'이 현재 LLM 구조의 토대가 되었다.<br>
`Transformer`는 전적으로 `Multi-head Self Attention`을 여러 층으로 쌓아 올린 구조이다. 이로 인해, 텍스트에 대해 계층적이고 점점 더 정교한 표현을 학습할 수 있다.

### 2.2. Transformer의 1단계
### Positional Encoding
입력 순서대로 토큰을 처리하는 `RNN`과 달리, 순서에 대한 고유한 개념 없이 병렬로 입력되는 `Self Attention Mechanism` 특성 상, 토큰마다 위치 정보를 부여해줘야 한다. <br>
주기성을 갖는 삼각함수 $\rm{sin}$과 $\rm{cos}$를 위치에 따라 번갈아 적용하며 토큰의 상대적 위치를 파악한다 (단어 순서를 모델이 처리할 벡터 자체에 인코딩하는 작업).

$$PE_{(pos,2i)}=\rm{sin}\it{(pos/1000^{2i/d})}$$
$$PE_{(pos,2i+1)}=\rm{cos}\it{(pos/1000^{2i/d})}$$

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 6.</b> Transformer의 Positional Encoding 주기성 시각화
</p>

### 2.3. Transformer의 2단계
### Transformer Block
다수의 레이어로 다층 구조를 이루고 있는 `Transformer` 블록의 구성:<br>
`Multi-head Self Attention`, `Feed-forward Layer`, `Residual Connection`, `Layer Normalization`

<b>Feed-forward Layer</b><br>
두 개의 `Linear Layer`로 구성되어 있으며, `Multi-head Self Ateention`의 출력을 새로운 차원으로 투영하는 역할을 수행<br>
구조적으로, `선형 변환` → `ReLU 활성화 함수` → `선형 변환`으로 이어진다.
$$FFN(x)=\rm{max}\it{(0,~xW_1+b_1)W_2+b_2}$$
- `Self Attention`에 비선형성을 추가 → 선형 결합만으로 표현할 수 없는 복잡한 패턴 포착 가능
- 병렬화 가능

<b>Residual Connections</b><br>
- 중간 레이어의 변환을 건너뛰고, 두 레이어 사이에서 정보를 직접 전달하는 방식<br>
- 레이어 간에 지름길을 제공해 기울기가 낮은 레이어까지 원활하게 전달되도록 도움
- 더 효율적이고 빠르게 학습할 수 있음

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 7.</b> Residual Connections의 효과
</p>

<b>Layer Normalization</b><br>
- 딥러닝 모델 내부의 은닉층 값들을 일정한 범위 안에 유지하여 학습을 안정적으로 만드는 정규화 기법
- `Batch Normalization`의 대안으로 제안된 방법
- 단일 벡터를 입력으로 받아 해당 벡터의 평균과 표준편차를 계산
- 이를 활용해 벡터를 정규화 및 스케일링

$$\mu=\frac{1}{d}\sum_{i=1}^d x_i ~~~\sigma=\sqrt{\frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2}$$
$$\hat{x}=\frac{(x-\mu)}{\sigma}$$
<br>
최종 변환 단계에서는 학습 과정에서 함께 학습되는 두 개의 파라미터 $\gamma$와 $\beta$를 활용
<br><br>

$$\rm{LayerNormalization}=\it{\gamma\hat{x}+\beta}$$

학습 과정에서 발생하는 출력 값 분포에 대한 변동성을 줄이기 위해 `Layer Normalization` 단계를 추가함으로써 기울기까지 정규화하는 효과를 얻는다.

<b>최종 블록 구조</b><br>
- PE: `Positional Encoding`
- MSA: `Multi-head Self Attention`
- LN: `Layer Normalization`
- FFN: `Feed-Forward Network`
- Add: `Residual Connections`
```
      ┌─────────────────────────┐  ┌─────────────┐
      │                         ↓  │             ↓
Input → Embedding → PE → MSA → Add → LN → FFN → Add → LN 
```
$$\mathbf{H}=\rm{LayerNorm(\mathbf{X}+MultHeadSelfAttention(\mathbf{X}))}$$
$$\mathbf{H}=\rm{LayerNorm(\mathbf{H}+FFN(\mathbf{H}))}$$
<b>참고 사항</b>
- 일부 아키텍처에서는 `LN`이 `FFN` 이전이 아니라 이후에 위치하기도 함. - 아직 논쟁중
- 최신 LLM에서 최대 96개의 Transformer 블록을 쌓아 사용. 블록 내부 구조는 거의 동일
- 절대적 PE는 시퀀스 초반 단어들을 과도하게 표현하는 결함 존재. 상대적 위치를 고려하는 다양한 변형 기법 제안

### 2.4. Cross Attention
`Transformer Decoder`에서는 `Encoder`와 같은 일련의 레이어들로 구성되어 있으나, `Self Attention`이 아닌 `Cross Attention`을 사용함

<b>Self Attention</b> : Query $=$ Key $=$ Value <br>
<b>Cross Attention</b> : Query $\ne$ Key, Value <br>
- <b>Query</b>: Decoder 의 이전 Layer 에서 전달되며, 현재 번역작업을 어디까지 진행하였는지 데이터를 가지고 있음. (current context) target 언어의 토큰값이 들어가게 됨.
- <b>Key</b>: Encoder 에서 Indexing 된 토큰들. (Source 언어의 값을 지님)
- <b>Value</b>: Encoder 에서 Indexing 된 토큰들. (Source 언어의 값을 지님)

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 8.</b> Cross Attention
</p>
<!-- https://cypsw.tistory.com/entry/Transformer-%EC%9D%98-CrossAttention -->

여기서 먼저 Key 와 Value 를 곱함으로서 Source 언어에서 단어들간 어떤 관계가 있는지 찾은 뒤, 해당 결과를 다시 Query 값과 곱한다. (이 때 Query 값에는 Target 언어 Token 값이 존재)

이 과정을 통해 Target 언어에 해당되는 Token 과 Source 언어에 해당되는 Token 의 연관성을 찾고,
다음에 올 단어를 예측해서 최종적인 번역 결과를 출력하는 것이다.

계산 과정은 `Self Attention`과 동일하며, Attention() 함수에서 $K$와 $V$ 자리에 자기 자신 벡터가 아닌 `Encoder`에서 넘어온 벡터를 넣기만 하면 그것이 `Cross Attention`.

### Why?
- 서로 다른 두 정보(입력값과 출력값) 사이의 연결 고리를 만들기 위함
- `Self Attention`이 문장 내부의 맥락을 파악한다면, `Cross Attention`은 외부의 정보 (`Encoder`가 준 힌트)를 가져와서 현재 내가 내뱉어야 할 정보와 대조하는 역할을 수행

### 2.5. Masked Attention
모델이 미래 의 정보를 미리 보지 못하게 하기 위해 `Decoder`의 첫 번째 `Self Attention`에 추가적인 Mask를 적용

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 9.</b> Masked Attention
</p>

## 3. 트랜스포머 학습하기
<b>핵심</b>: 복잡한 패턴을 유연하게 포착하는 관계를 스스로 학습 <br>

초기의 `Transformer` 모델은 `Encoder`와 `Decoder`가 모두 존재하는 seq2seq 구조였으나, 이후 등장한 LLM은 주로 `Decoder`만 존재하며 주어진 데이터로부터 스스로 라벨을 만들어 학습하는 `Self-supervised` 방식의 언어 모델로 학습되었다.

-- 주어진 단어들의 문맥을 고려하여 다음 단어 $x$가 나올 확률 $p(w|h)$를 추정하는 방식
$$P(w|h)=P(w_n|w_{1:n-1})=\prod_{i=1}^nP(w_i|w_{i:i-1})$$
- $w$: &emsp;'다음 단어' 또는 '현재 시점의 단어'
- $h$, $w_{1:n-1}$: &emsp;$w$ 등장 이전 단어들의 시퀀스
- $P(w|h)$: &emsp;$h$가 주어졌을 때 $w$가 등장할 조건부 확률

```
선형 투영 레이어 → 소프트맥스 레이어 → 출력
```
<b>선형 투영 레이어</b>: $D$ 차원의 결과물을 전체 $V$ (어휘 수)만큼의 긴 확률 벡터로 늘리는 역할을 수행. `Embedding`을 거꾸로 돌린다하여 `Unembedder`라 불림

### 3.1. Cross Entropy Loss
주석이 없는 텍스트 코퍼스를 활용하여 모델이 예측한 다음 단어의 확률과 실제 다음 단어의 확률 간의 차이를 최소화하는 방향으로 학습하기 위해, <b>교차 엔트로피 손실 (Cross Entropy Loss)</b> 손실 함수를 사용한다. 이는 예측된 확률 분포와 실제 확률 분포 간의 차이를 계산한다. 예측된 확률 분포는 확률 벡터이며, 실제 확률 분포는 해당 시퀀스에서 다음에 올 단어에 해당하는 위치만 1이고 나머지는 0인 원-핫 벡터이다.

$$\mathcal{L}_{CE}=-\sum_{i=1}^{C} y_t[w]~\mathrm{log}~\hat{y}_t[w]$$

최종 손실은 시퀀스 전체에 대해 계산한 손실 평균으로 구한다.

### 3.2. Greedy Decoding
확률 벡터를 얻은 후, 주어진 문맥으로부터 다음에 등장할 단어를 예측할 때 가장 높은 확률을 가진 단어를 선택할 수 있는데, 이를 단순히 확률이 가장 높은 단어를 선택한다고 하여 그리디 디코딩이라고 한다.

<b>핵심</b>: 각 단계에서 국소적으로 가장 최선이라고 판단되는 단어(가장 높은 확률)를 선택 <br>

$$w_t=\rm{argmax}\it{_{w\in V}P(w|w_{<t})}$$

결과 예측이 너무 쉽고 정형화되었으며, 반복적인 경향이 있어 잘 사용하지 않는다.

### 3.3. 자기회귀 생성, 인과 언어 모델링
그리디 디코딩은 주어진 문맥에서 다음 단어를 고를 때, 단순히 확률이 가장 높은 단어를 고르는 방식이 생각보다 완성된 문장의 질이 좋지 않은 경우도 많다. <br>
더 정교하고 비결정적인 샘플링 기법을 사용하는데, 해당 샘플링 과정을 `디코딩` 이라고 부르며, `자기회귀 생성` 또는 `인과 언어 모델링`이라고도 한다.
- 랜덤 샘플링: 소프트맥스를 통해 계산된 전체 단어의 확률 분포에 따라 무작위로 다음 단어를 선택하는 방식. 특정 단어의 확률이 낮더라도 선택될 가능성이 열려 있어 생성되는 문장의 다양성이 매우 높아지지만, 때로는 문맥과 전혀 상관없는 엉뚱한 단어가 선택될 위험이 있음.
- Top-k 샘플링: 확률이 높은 순서대로 상위 $k$개의 단어 후보군을 먼저 거른 뒤, 그 안에서만 확률 분포에 따라 샘플링을 수행. $k$ 이외의 낮은 확률을 가진 단어(Long Tail)들을 원천 차단함으로써 문장의 일관성을 높이고 엉뚱한 대답이 나오는 것을 방지할 있음.
- Top-p 샘플링: 상위 $k$개처럼 개수를 고정하는 대신, 확률의 누적 합계가 설정값 $p$에 도달할 때까지의 후보군을 동적으로 선택함. 문맥에 따라 고려해야 할 단어의 후보 수가 유동적으로 변하므로, Top-k 방식보다 더 유연하고 자연스러운 문장 생성이 가능.
- 온도 샘플링: 확률 분포의 '뾰족함'을 조절하는 매개변수 $T$를 사용하여 모델의 창의성을 제어. 온도를 높이면($T > 1$) 확률 분포가 평평해져 더 다양하고 창의적인 답변이 나오고, 온도를 낮추면($T < 1$) 확률이 높은 단어에 더 집중하게 되어 결정론적이고 안정적인 문장을 생성.

### 3.4. 미지 단어 문제
실제로 모델이 학습된 이후 알지 못하는 단어를 만나게 되면 특별한 토큰 \<UNK>를 할당한다. 예를 들어, 학습 데이터에 'big', 'bigger', 'small'은 포함되어 있고, 'smaller'는 없다면, 해당 단어를 모르는 단어로 간주하고 \<UNK> 토큰으로 처리한다.

#### 3.4.1. 해결책
토크나이저가 형태소와 문법 규칙을 인식할 수 있다면 더욱 정교한 처리와 일반화가 가능하다. \<UNK> 토큰을 피할 수 있는 방법 중 하나는 텍스트를 서브워드 단위로 나누는 것이다.

#### 3.4.2. 바이트 페어 인코딩 (BPE; Byte-pair Encoding)

<b>핵심</b>: 자주 쓰이는 단어는 하나의 토큰으로, 드문 단어는 의미 있는 단위(서브워드)로 쪼개어 관리 
- 모든 단어를 문자(Character) 단위로 분해.
- 데이터 전체에서 가장 자주 함께 나타나는 문자 쌍(Pair)을 탐색.
- 해당 쌍을 하나의 새로운 유닛으로 병합(Merge)하고 사전에 추가.
- 정해진 사전 크기(Vocabulary Size)에 도달할 때까지 이 과정을 반복.

#### 3.4.3. Why?
<b>신조어 및 OOV(Out-Of-Vocabulary) 문제 해결</b>: 어휘집에 없는 모르는 단어가 나와도 문자 단위로 쪼개져 있으므로 \<UNK> 토큰으로 처리하지 않고 어떻게든 의미 있는 단위로 해석할 수 있다.

<b>효율적인 어휘집 크기 유지</b>: 단어 전체를 저장하는 방식보다 훨씬 적은 수의 토큰만으로도 방대한 양의 텍스트를 표현할 수 있어 계산 효율성이 높다.

<b>정보의 압축</b>: 자주 등장하는 패턴(예: '-ing', 'pre-')을 하나의 토큰으로 묶음으로써 모델이 언어의 구조를 더 잘 이해하도록 돕는다.

## 4. 마스크드 언어 모델링 탐구하기
### 4.1. BERT (Bidirectional Encoder Representations from Transformers)
<b>핵심</b>: Transformer의 인코더(Encoder) 구조를 활용하여, 문맥을 양방향으로 동시에 이해하도록 설계된 대표적인 양방향 언어 모델. 텍스트 생성 작업을 크게 전제하지 않았다는 것을 의미. 

<b>양방향 인코더</b>: 문장을 왼쪽에서 오른쪽으로만 읽는 기존의 인과 언어 모델과 달리, 문장 전체를 한꺼번에 입력받아 각 단어의 좌우 문맥을 동시에 참조.

```
Input: 나는 오늘 영화를 봤다
Process: 영화를
Reference: '나는 오늘', '봤다' (양방향)
```
한계점: 이전 문맥만 보고 다음 단어를 예측하는 방식인 인과 언어 모델로는 더 이상 학습할 수 없고, 양방향 인코더 모델을 훈련하기 위해서 새로운 학습 방식이 필요하다.

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 10.</b> 인과 언어 모델과 양방향 언어 모델의 차이
</p>

<b>마스크드 언어 모델</b>: 학습 시 입력 문장의 일부 단어를 \<MASK> 토큰으로 가리고, 주변 문맥만을 이용해 가려진 단어를 예측하는 방식으로 학습한다. 즉, 전체 문맥이 주어진 상태에서 BERT는 \<MASK>라는 특별한 토큰으로 가려진 토큰을 예측해야 한다.
```
Input: 나는 오늘 <MASK>를 봤다.
Process: <MASK>
Reference: '나는 오늘', '봤다'
Prediction: '영화' or '드라마' 같은 단어가 들어가야 함
```

원 논문에서는토큰의 15%를 무작위 마스킹. 오른쪽 단어들만 마스킹하는 인과 언어 모델과는 다르게 문장 전체에서 골고루 가린다는 의미

#### 그 외 다양한 토큰
- [CLS]: 입력의 시작
- [SEP]: 입력 내 문장

2024년 이전까지는 bert와 같은 모델이 아예 텍스트를 생성할 수 없다고 여겨졌는데, 2024년에 발표된 두 편의 연구에서 BERT 계열 모델도 적절히 변형하면 텍스트 생성을 수행할 수 있다는 사실이 밝혀졌다. 

<p align="center">
  <img src="about:blank" width=600><br>
  <b>Figure 11.</b> 마스크드 언어 모델을 활용한 텍스트 생성
</p>

그러나, BERT는 본래 인과 언어 모델이 아니기에, 아주 긴 글을 처음부터 끝까지 생성할 때는 GPT보다 일관성이 떨어질 수 있다.

## 5. 내부 메커니즘 시각화하기


## 6. 트랜스포머 활용하기


## Reference
[1] https://github.com/ai-agent-kr/Modern-AI-Agents/tree/main/ch02


<!-- Code Section -->

```

```