from pathlib import Path
import pandas as pd

def load_ott(raw_path: str | Path) -> pd.DataFrame:
    """
    Load and normalize Ott et al. (deceptive opinion spam) dataset.
    
    Target schema:
      - text: string
      - source_dataset: 'ott'
      - authorship_label: 'human' (all human written)
      - deception_label: 'genuine' (truthful) | 'deceptive' (deceptive)
      - domain: 'hotel'
      - original_id: str
    """
    df = pd.read_csv(raw_path)
    
    # Text normalization
    df['text'] = df['text'].astype(str).str.strip().apply(lambda s: ' '.join(s.split()))
    
    # Map deception
    deception_map = {
        'truthful': 'genuine',
        'deceptive': 'deceptive',
    }
    df['deception_label'] = df['deceptive'].map(deception_map)
    df['authorship_label'] = 'human'
    df['source_dataset'] = 'ott'
    df['domain'] = 'hotel'
    df['source_id'] = [f"ott_{i}" for i in range(len(df))]
    df['original_id'] = df['source_id']
    
    cols = ['original_id', 'source_id', 'text', 'source_dataset', 'authorship_label', 'deception_label', 'domain']
    return df[cols].copy()

