# TransE from Scratch

*Translating Embeddings for Modeling Multi-relational Data* (NeurIPS, 2013)
의 알고리즘을 PyTorch로 직접 구현한 코드입니다.

## 구현 내용

- **초기화**: $Unif(-6/\sqrt{k}, 6/\sqrt{k})$ 로 entity/relation 임베딩 초기화, relation은 초기화 직후 정규화
- **학습 루프**: 매 epoch마다 entity normalize → mini-batch 샘플링 → negative sampling (corrupted triplet) → margin-based ranking loss
- **평가**: Link prediction (raw / filtered 두 세팅), Mean Rank, MRR, Hits@10

## 데이터셋

[FB15k-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312) — Freebase 기반 표준 KGE 벤치마크.
원본 FB15k에서 head-tail을 뒤집은 역방향 관계(`!/people/person/nationality` 류)를 제거한 버전 (논문에서의 동일한 데이터 전처리 과정)

## 실행 방법

### Colab에서 바로 실행
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0hSeunghwan/TransE_implementation/blob/main/TransE.ipynb)


### 로컬에서 실행
```bash
pip install torch jupyter
jupyter notebook TransE.ipynb
```

## 주요 하이퍼파라미터 (FB15k 원 논문 세팅 참고)

| 파라미터 | 값 |
|---|---|
| 임베딩 차원 $k$ | 50 |
| Margin $\gamma$ | 1.0 |
| 거리 함수 | L1 |
| Learning rate | 0.01 |
| Optimizer | SGD |

## 참고문헌

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013).
Translating Embeddings for Modeling Multi-relational Data. *NeurIPS*.

Toutanova, K., & Chen, D. (2015). Observed versus Latent Features for Knowledge Base
and Text Inference. *CVSC Workshop*. (FB15k-237 데이터셋 출처)

## License

MIT
