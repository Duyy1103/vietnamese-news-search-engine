import pandas as pd
import os

def extract_error_cases():
    rankings_file = "data/evaluation/models/full-16/ranking_details_16methods.csv"
    qrels_file = "data/evaluation/models/full-16/qrels_16methods.csv"
    output_file = "data/evaluation/models/full-16/error_cases_16methods.csv"

    # Read data
    rankings = pd.read_csv(rankings_file)
    qrels = pd.read_csv(qrels_file)

    # Filter only top 10
    top10_rankings = rankings[rankings['rank'] <= 10].copy()

    # Create a fast lookup for relevance: (query_id, str(doc_id)) -> relevance
    qrels['doc_id'] = qrels['doc_id'].astype(str)
    top10_rankings['doc_id'] = top10_rankings['doc_id'].astype(str)
    
    qrels_dict = set(zip(qrels[qrels['relevance'] >= 1]['query_id'], qrels[qrels['relevance'] >= 1]['doc_id']))

    # We will identify error cases
    error_cases = []

    # Group by method and query_id
    for (method, query_id), group in top10_rankings.groupby(['method', 'query_id']):
        group = group.sort_values('rank')
        if len(group) == 0:
            continue
            
        # Get rank 1 document
        rank1_doc = group.iloc[0]
        rank1_id = str(rank1_doc['doc_id'])
        
        # Check if rank 1 is relevant
        is_rank1_relevant = (query_id, rank1_id) in qrels_dict
        
        if not is_rank1_relevant:
            # Rank 1 is an error. Check if there are ANY relevant docs in top 10
            any_relevant_in_top10 = any((query_id, str(doc_id)) in qrels_dict for doc_id in group['doc_id'])
            
            error_type = 'top_result_not_relevant' if any_relevant_in_top10 else 'no_relevant_in_top_k'
            
            # Prepare row for error_cases
            row_dict = rank1_doc.to_dict()
            row_dict['relevance'] = 0
            row_dict['error_type'] = error_type
            error_cases.append(row_dict)

    # Create DataFrame and save
    if error_cases:
        error_df = pd.DataFrame(error_cases)
        # Reorder columns to match requested format if possible
        expected_cols = ['method', 'query_id', 'query', 'query_topic', 'rank', 'doc_id', 'score', 'title', 'snippet', 'source', 'topic', 'url', 'chunk_id', 'relevance', 'error_type']
        # Add missing columns if any
        for col in expected_cols:
            if col not in error_df.columns:
                error_df[col] = ''
        
        # Select existing columns that match expected
        final_cols = [c for c in expected_cols if c in error_df.columns]
        error_df = error_df[final_cols]
        
        error_df.to_csv(output_file, index=False)
        print(f"Successfully extracted {len(error_df)} error cases to: {output_file}")
    else:
        print("No error cases found! (This means every method has a relevant document at Rank 1 for all queries, which is unlikely).")

if __name__ == "__main__":
    extract_error_cases()
