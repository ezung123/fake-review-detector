from pathlib import Path
import pandas as pd

def load_salminen(raw_path: str | Path) -> pd.DataFrame:
    """
    Load and normalize Salminen et al. fake reviews dataset.
    
    Target schema:
      - text: string
      - source_dataset: 'salminen'
      - authorship_label: 'human' (OR) | 'ai' (CG)
      - deception_label: 'unknown' (never map AI to deceptive)
      - domain: 'e-commerce'
      - original_id: str
    """
    df = pd.read_csv(raw_path)
    
    # Text normalization
    df['text'] = df['text_'].astype(str).str.strip().apply(lambda s: ' '.join(s.split()))
    
    # Authorship mapping: OR = Original (human), CG = Computer Generated (ai)
    authorship_map = {
        'OR': 'human',
        'CG': 'ai',
    }
    df['authorship_label'] = df['label'].map(authorship_map)
    df['deception_label'] = 'unknown'
    df['source_dataset'] = 'salminen'
    df['domain'] = 'e-commerce'
    df['source_id'] = [f"salminen_{i}" for i in range(len(df))]
    df['original_id'] = df['source_id']
    
    cols = ['original_id', 'source_id', 'text', 'source_dataset', 'authorship_label', 'deception_label', 'domain']
    return df[cols].copy()

