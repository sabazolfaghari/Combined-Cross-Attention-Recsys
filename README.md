# Combined-Cross-Attention-Recsys
This code is the combination of Spatiotemporal GCN and Trend-aware TGN recommender systems with the cross attention mechanism. This is done in order to consider both temporal aspects of user-item interaction and the changes in item popularity. The combined embeddings are then passed through an MLP to predict user–item ratings.

The framework is designed for research and experimentation on recommender systems where multiple embedding sources need to be integrated.

## Dataset
This project is based on the MovieLens 100K dataset, a widely used benchmark for recommender systems with 100,000 ratings.

We use the raw MovieLens ratings to build both ground truth ratings (actual user–movie interactions) and embeddings generated from two different models:

- Spatiotemporal GCN model → Captures only time-aware patterns from interaction history of user-item pairs.

- Trend-aware TGN model → captures time-aware interactions which are also embedded with rich movie metadata.

## Installation
This project is implemented with Python 3.11.0.

Install dependencies with pip install.

## Usage
```bash
python -u "combined.py"
