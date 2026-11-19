from pathlib import Path
import pandas as pd
import numpy as np

def load_raid(raw_path: str | Path, seed: int = 42) -> pd.DataFrame:
    """
    Load, filter, and sample RAID reviews dataset.
    
    Constraints:
      - Filter: domain == 'reviews' and attack == 'none'
      - Keep all human rows
      - Sample AI rows to ~2x human count, spread evenly across all 11 models
      - Set domain = 'movie' (confirmed all titles are films)
      - authorship_label: 'human' if model == 'human' else 'ai'
      - deception_label: 'unknown'
      - Preserve source_id to prevent prompt/generation data leakage across splits
    """
    df = pd.read_csv(raw_path)
    
    # Filter domain == 'reviews' and attack == 'none'
    df = df[(df['domain'] == 'reviews') & (df['attack'] == 'none')].copy()
    
    # Text normalization
    df['text'] = df['generation'].astype(str).str.strip().apply(lambda s: ' '.join(s.split()))
    
    # Split human vs AI
    human_df = df[df['model'] == 'human'].copy()
    ai_df = df[df['model'] != 'human'].copy()
    
    # Calculate target AI samples: 2x human count
    n_human = len(human_df)
    target_ai = n_human * 2 # 1886
    
    models = sorted(ai_df['model'].unique())
    n_models = len(models)
    base_per_model = target_ai // n_models
    remainder = target_ai % n_models
    
    sampled_ai_list = []
    for idx, model_name in enumerate(models):
        m_count = base_per_model + (1 if idx < remainder else 0)
        m_subset = ai_df[ai_df['model'] == model_name]
        sampled_subset = m_subset.sample(n=m_count, random_state=seed)
        sampled_ai_list.append(sampled_subset)
        
    sampled_ai = pd.concat(sampled_ai_list, ignore_index=True)
    combined = pd.concat([human_df, sampled_ai], ignore_index=True)
    
    combined['authorship_label'] = np.where(combined['model'] == 'human', 'human', 'ai')
    combined['deception_label'] = 'unknown'
    combined['source_dataset'] = 'raid'
    combined['domain'] = 'movie'
    combined['original_id'] = combined['id'].astype(str)
    
    cols = ['original_id', 'source_id', 'text', 'source_dataset', 'authorship_label', 'deception_label', 'domain', 'model']
    return combined[cols].copy()

