## Demo App
![demo1](README_files/demo1.png)
![demo2](README_files/demo2.png)

## System Architecture
![System Architecture](README_files/System_Architecture_hyperlinked.svg)

## Evaluation Results
![evaluation_results](README_files/Evaluation_Results.png)

# Disclaimer
- Precision@3 wasn't printed because in a one-to-one matching scenario—where each generated image corresponds to exactly one ground truth image—Precision@3 becomes redundant. In this context, if the correct match is in the top 3, both Precision@3 and Recall@3 would reflect a "hit." Thus, we focus on Recall@3 (and MRR) to measure retrieval performance without adding redundant metrics.