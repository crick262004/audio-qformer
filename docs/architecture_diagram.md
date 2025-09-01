# Audio Q-Former Architecture Diagram

## High-Level System Architecture

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Audio      │    │   SeamlessM4T   │    │  Q-Former   │    │  Projector  │    │     LLM     │
│  Input      │───▶│   Encoder       │───▶│             │───▶│             │───▶│  (Llama)    │
│  (.wav)     │    │                 │    │             │    │             │    │             │
└─────────────┘    └─────────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                           │                       │                                      │
                           │                       │                                      │
                           ▼                       ▼                                      ▼
                  ┌─────────────────┐    ┌─────────────────┐                   ┌─────────────┐
                  │ Audio Features  │    │ Query Tokens    │                   │ Text Output │
                  │ [B, T, D]       │    │ [B, Q, D]       │                   │ Summary     │
                  └─────────────────┘    └─────────────────┘                   └─────────────┘
                           │                       │                                      
                           └───────────────────────┼──────────────────────────────────────┘
                                                   │                                      
                                                   ▼                                      
                                          ┌─────────────────┐                           
                                          │ Contrastive     │                           
                                          │ Learning        │                           
                                          │ (InfoNCE +      │                           
                                          │  Matching)      │                           
                                          └─────────────────┘                           
```

## Detailed Q-Former Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    Q-Former                             │
                    │                                                         │
    Audio Features  │  ┌───────────────┐                                     │  Query 
    [B, T, D] ─────▶│  │ Query Tokens  │                                     │  Outputs
                    │  │ [B, Q, D]     │                                     │  [B, Q, D]
                    │  └───────┬───────┘                                     │    │
                    │          │                                             │    │
                    │          ▼                                             │    │
                    │  ┌───────────────┐    ┌─────────────────────────────┐  │    │
                    │  │ Layer Norm    │    │       Cross-Attention       │  │    │
                    │  │ (Queries)     │    │                             │  │    │
                    │  └───────┬───────┘    │  Q: Normalized Queries      │  │    │
                    │          │            │  K,V: Normalized Audio      │  │    │
                    │          │            │                             │  │    │
    Audio Features  │          │            │  Multi-Head Attention       │  │    │
    [B, T, D] ──────┼──────────┼───────────▶│  + Residual Connection      │  │    │
                    │          │            │                             │  │    │
                    │          ▼            └─────────────┬───────────────┘  │    │
                    │  ┌───────────────┐                  │                  │    │
                    │  │ Layer Norm    │                  ▼                  │    │
                    │  │ (Audio)       │          ┌───────────────┐          │    │
                    │  └───────────────┘          │ Feed-Forward  │          │    │
                    │                             │ Network       │          │    │
                    │                             │ (MLP + GELU)  │          │    │
                    │                             └───────┬───────┘          │    │
                    │                                     │                  │    │
                    │                                     ▼                  │    │
                    │                             ┌───────────────┐          │    │
                    │                             │ Final Layer   │          │    │
                    │                             │ Norm          │          │    │
                    │                             └───────┬───────┘          │    │
                    │                                     │                  │    │
                    └─────────────────────────────────────┼──────────────────┘    │
                                                          │                       │
                                                          └───────────────────────┘
```

## Training Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Audio File  │────▶│ Load Audio  │────▶│ SeamlessM4T │────▶│ FilterBank  │
│ (.wav/.mp3) │     │ (16kHz)     │     │ Converter   │     │ Features    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Loss        │◀────│ LLM Forward │◀────│ Projector   │◀────│ Q-Former    │
│ Computation │     │ Pass        │     │ Layer       │     │ Processing  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Loss Components                                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │ LM Loss     │    │ Contrastive     │    │ Matching Loss               │  │
│  │             │    │ Loss (InfoNCE)  │    │                             │  │
│  │ Cross-      │ +  │                 │ +  │ Binary Classification       │  │
│  │ Entropy     │    │ Audio ↔ Text    │    │ Positive vs Negative Pairs  │  │
│  │             │    │ Alignment       │    │                             │  │
│  └─────────────┘    └─────────────────┘    └─────────────────────────────┘  │
│                                                                             │
│  λ_lm = 1.0         λ_con = 0.2              λ_match = 0.1                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Contrastive Learning Framework

```
                        Audio Features        Text Features
                        [B, D_audio]         [B, D_text]
                              │                    │
                              ▼                    ▼
                    ┌─────────────────┐  ┌─────────────────┐
                    │ Audio Projector │  │ Text Projector  │
                    │ (MLP)           │  │ (MLP)           │
                    └─────────┬───────┘  └─────────┬───────┘
                              │                    │
                              ▼                    ▼
                        Audio Embeddings     Text Embeddings
                        [B, D_contrast]      [B, D_contrast]
                              │                    │
                              └────────┬───────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │  Similarity Matrix  │
                            │  [B, B+K] where     │
                            │  K = queue size     │
                            └─────────┬───────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │    InfoNCE Loss     │
                            │                     │
                            │ Positive: (i,i)     │
                            │ Negatives: (i,j≠i)  │
                            │           + Queue   │
                            └─────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │                    Feature Queues                               │
    │                                                                 │
    │  Audio Queue                    Text Queue                      │
    │  ┌─────────────┐               ┌─────────────┐                 │
    │  │ [K, D_con]  │               │ [K, D_con]  │                 │
    │  │ FIFO Buffer │               │ FIFO Buffer │                 │
    │  │             │               │             │                 │
    │  │ Maintains   │               │ Maintains   │                 │
    │  │ diverse     │               │ diverse     │                 │
    │  │ negatives   │               │ negatives   │                 │
    │  └─────────────┘               └─────────────┘                 │
    └─────────────────────────────────────────────────────────────────┘
```

## Inference Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Audio Input │────▶│ Feature     │────▶│ Q-Former    │────▶│ Projector   │
│             │     │ Extraction  │     │ Encoding    │     │ Layer       │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Text        │◀────│ LLM         │◀────│ Instruction │◀────│ Audio       │
│ Summary     │     │ Generation  │     │ Embedding   │     │ Embeddings  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                            │
                            ▼
                    ┌─────────────┐
                    │ Beam Search │
                    │ Decoding    │
                    │ (max_tokens)│
                    └─────────────┘
```