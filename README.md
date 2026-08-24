# TenRec RGCN Popularity Bias

Research project for diagnosing and later mitigating popularity-bias transfer in cross-domain recommendation.

## Frozen baseline: Baseline_Diagnosis_v1_frozen

Current pipeline:

canonical TenRec  
-> shared warm-start users  
-> 8 relation types  
-> multi-relational FullGraph RGCN  
-> user/item embeddings  
-> BPR-only training  
-> Top-K recommendation  
-> post-hoc popularity diagnosis

## Relations

0. click
1. like
2. share
3. follow
4. rev_click
5. rev_like
6. rev_share
7. rev_follow

## Training

The baseline uses BPR only.

Popularity is NOT used in:
- training loss
- reranking
- model input

## Popularity diagnosis

Item popularity is defined as:

Pop(i) = number of unique TRAINING users who clicked item i

Current diagnosis metrics include:
- Recall@10
- NDCG@10
- ARP@10
- Profile Popularity
- Popularity Lift
- Head Ratio
- Tail Ratio
- Tail Coverage
- Catalog Coverage

## Current scope

This frozen baseline does NOT yet include:
- PopReg
- cross-domain transfer
- MMD
- conditional MMD
- adaptive regularization
- VAE

QK-video and QK-article use the same warm-start shared-user cohort.
Validation and test targets retain at least one training click.
