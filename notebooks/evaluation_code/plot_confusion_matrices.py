import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set visual style
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300

def parse_method_name(method):
    if 'cfg_' in method:
        family, config = method.split('cfg_', 1)
    elif '_' in method:
        family, config = method.split('_', 1)
    else:
        family = method
        config = 'unknown'
        
    if family == 'semantic': family = 'Semantic (E5)'
    elif family == 'hybrid': family = 'Hybrid'
    elif family == 'bm25': family = 'BM25'
    elif family == 'tfidf': family = 'TF-IDF'
    
    config_map = {
        'accented_with_stopwords': 'Accented + Stopwords',
        'accented_no_stopwords': 'Accented (No Stopwords)',
        'unaccented_with_stopwords': 'Unaccented + Stopwords',
        'unaccented_no_stopwords': 'Unaccented (No Stopwords)'
    }
    return f"{family}\n({config_map.get(config, config)})"

def plot_confusion():
    rankings_file = "data/evaluation/models/full-16/ranking_details_16methods.csv"
    qrels_file = "data/evaluation/models/full-16/qrels_16methods.csv"
    out_dir = "data/evaluation/models/full-16/analysis_plots"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    # Assume script is run from project root. If run from tai_evaluation_code, adjust paths.
    if not os.path.exists(rankings_file):
        # Adjusted for relative paths if script run from subdirectory
        rankings_file = "../../data/evaluation/models/full-16/ranking_details_16methods.csv"
        qrels_file = "../../data/evaluation/models/full-16/qrels_16methods.csv"
        out_dir = "../../data/evaluation/models/full-16/analysis_plots"
        os.makedirs(out_dir, exist_ok=True)

    rankings = pd.read_csv(rankings_file)
    qrels = pd.read_csv(qrels_file)

    # Filter top 10
    rankings = rankings[rankings['rank'] <= 10]

    # Map relevance: >=1 is Relevant (Positive), 0 is Irrelevant (Negative)
    qrels['is_relevant'] = qrels['relevance'] >= 1
    
    methods = rankings['method'].unique()

    for method in methods:
        method_rankings = rankings[rankings['method'] == method]
        retrieved_set = set(zip(method_rankings['query_id'], method_rankings['doc_id'].astype(str)))
        
        # Initialize Confusion Matrix counts
        TP = FP = FN = TN = 0
        
        for index, row in qrels.iterrows():
            q_id = str(row['query_id'])
            d_id = str(row['doc_id'])
            is_rel = row['is_relevant']
            
            is_retrieved = (q_id, d_id) in retrieved_set
            
            if is_retrieved and is_rel:
                TP += 1
            elif is_retrieved and not is_rel:
                FP += 1
            elif not is_retrieved and is_rel:
                FN += 1
            elif not is_retrieved and not is_rel:
                TN += 1

        # Create Confusion Matrix
        cm = np.array([[TN, FP],
                       [FN, TP]])
                       
        plt.figure(figsize=(6, 5))
        ax = plt.gca()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Not Retrieved', 'Retrieved (Top 10)'],
                    yticklabels=['Not Relevant', 'Relevant'],
                    annot_kws={"size": 16})
        
        ax.set_title(parse_method_name(method), fontsize=16, pad=15)
        ax.set_xlabel('Predicted (System Retrieval)', fontsize=12)
        ax.set_ylabel('Actual (Ground Truth)', fontsize=12)

        plt.tight_layout()
        out_path = os.path.join(out_dir, f'confusion_matrix_{method}.png')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Generated confusion matrix plot: {out_path}")

if __name__ == "__main__":
    plot_confusion()
