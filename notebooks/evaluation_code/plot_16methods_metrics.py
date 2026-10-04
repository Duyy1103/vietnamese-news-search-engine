import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

def parse_method_name(method):
    # e.g., semanticcfg_accented_with_stopwords
    if 'cfg_' in method:
        family, config = method.split('cfg_', 1)
    elif '_' in method:
        family, config = method.split('_', 1)
    else:
        family = method
        config = 'unknown'
        
    # Simplify family names
    if family == 'semantic': family = 'Semantic (E5)'
    elif family == 'hybrid': family = 'Hybrid'
    elif family == 'bm25': family = 'BM25'
    elif family == 'tfidf': family = 'TF-IDF'
    
    # Beautify config names
    config_map = {
        'accented_with_stopwords': 'Accented + Stopwords',
        'accented_no_stopwords': 'Accented (No Stopwords)',
        'unaccented_with_stopwords': 'Unaccented + Stopwords',
        'unaccented_no_stopwords': 'Unaccented (No Stopwords)'
    }
    config_pretty = config_map.get(config, config)
    
    return family, config_pretty

def plot_all():
    input_file = "data/evaluation/models/full-16/metrics_16methods.csv"
    out_dir = "data/evaluation/models/full-16/analysis_plots"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(input_file)
    
    # Parse families and configs
    parsed = df['method'].apply(parse_method_name)
    df['Family'] = [x[0] for x in parsed]
    df['Config'] = [x[1] for x in parsed]

    # Sort dataset by NDCG@10
    df = df.sort_values('ndcg@10', ascending=False)
    
    palette = sns.color_palette("Set2", 4)
    family_order = ['Semantic (E5)', 'Hybrid', 'BM25', 'TF-IDF']
    config_order = [
        'Accented + Stopwords', 
        'Accented (No Stopwords)', 
        'Unaccented + Stopwords', 
        'Unaccented (No Stopwords)'
    ]

    # 1. Overall NDCG@10 (Horizontal Bar Chart)
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df, x='ndcg@10', y='method', hue='Family', dodge=False, palette='viridis')
    plt.title('Overall NDCG@10 Ranking (16 Methods)', fontsize=18, pad=20)
    plt.xlabel('NDCG@10', fontsize=14)
    plt.ylabel('Method Configuration', fontsize=14)
    plt.xlim(0, 1.0)
    plt.legend(title='Method Family', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overall_ndcg10.png'))
    plt.close()

    # 2. Overall MRR@10 (Horizontal Bar Chart)
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df.sort_values('mrr@10', ascending=False), x='mrr@10', y='method', hue='Family', dodge=False, palette='mako')
    plt.title('Overall MRR@10 Ranking (16 Methods)', fontsize=18, pad=20)
    plt.xlabel('MRR@10', fontsize=14)
    plt.ylabel('Method Configuration', fontsize=14)
    plt.xlim(0, 1.05)
    plt.legend(title='Method Family', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overall_mrr10.png'))
    plt.close()

    # 3. Grouped Bar Chart by Family (NDCG@10)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df, x='Family', y='ndcg@10', hue='Config', order=family_order, hue_order=config_order, palette='Spectral')
    plt.title('NDCG@10 Comparison Across Families & Configurations', fontsize=18, pad=20)
    plt.xlabel('Method Family', fontsize=14)
    plt.ylabel('NDCG@10', fontsize=14)
    plt.legend(title='Text Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontsize=10, rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'grouped_ndcg10_by_family.png'))
    plt.close()

    # 4. Grouped Bar Chart by Family (Precision@10)
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(data=df, x='Family', y='precision@10', hue='Config', order=family_order, hue_order=config_order, palette='coolwarm')
    plt.title('Precision@10 Comparison Across Families & Configurations', fontsize=18, pad=20)
    plt.xlabel('Method Family', fontsize=14)
    plt.ylabel('Precision@10', fontsize=14)
    plt.legend(title='Text Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'grouped_precision10_by_family.png'))
    plt.close()

    # 5. Overall Latency Comparison
    df_latency = df.sort_values('mean_latency_seconds', ascending=False)
    plt.figure(figsize=(14, 10))
    sns.barplot(data=df_latency, x='mean_latency_seconds', y='method', hue='Family', dodge=False, palette='magma')
    plt.title('Mean Latency Comparison (Seconds/Query)', fontsize=18, pad=20)
    plt.xlabel('Mean Latency (Seconds)', fontsize=14)
    plt.ylabel('Method Configuration', fontsize=14)
    plt.legend(title='Method Family', loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'overall_latency.png'))
    plt.close()

    # 6. Scatter Plot: Trade-off (Latency vs NDCG@10)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='mean_latency_seconds', y='ndcg@10', hue='Family', style='Config', s=200, palette='bright')
    plt.title('Performance Trade-off: Accuracy vs Latency', fontsize=18, pad=20)
    plt.xlabel('Mean Latency (Seconds/Query)', fontsize=14)
    plt.ylabel('NDCG@10', fontsize=14)
    
    # Add labels to points (only families to avoid clutter)
    for i in range(df.shape[0]):
        plt.text(x=df['mean_latency_seconds'].iloc[i]+0.05, y=df['ndcg@10'].iloc[i], s=df['Family'].iloc[i], 
                 fontdict=dict(color='gray',size=9))

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'scatter_latency_vs_ndcg.png'))
    plt.close()
    
    print(f"Generated 6 charts in {out_dir}")

if __name__ == "__main__":
    plot_all()
