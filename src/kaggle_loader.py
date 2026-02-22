import pandas as pd
import json
from datetime import datetime
from src.data_collector import Market, Condition

def load_kaggle_markets(csv_path: str, limit: int = 1000, topic_filter: str = None) -> list[Market]:
    """
    Load markets from the Kaggle polymarket_markets.csv dataset.
    Maps CSV columns to the internal Market and Condition dataclasses.
    """
    df = pd.read_csv(csv_path)
    
    # Filter to only active and unclosed markets
    df = df[df['active'] == True]
    df = df[df['closed'] == False]
    
    # Require some volume but handle NaNs
    df['volume'] = df['volume'].fillna(0)
    df = df[df['volume'] > 5000] # Lowered to 5000 to find enough active pairs
    
    # Filter out markets where max price is 1 or min price is 0 (already resolved)
    df = df[df['outcomePrices'].notna()]
    df = df[~df['outcomePrices'].str.contains(r'1\.0|0\.0', regex=True)]
    
    if topic_filter:
        df = df[df['category'].str.lower() == topic_filter.lower()]
        
    markets = []
    
    for _, row in df.head(limit).iterrows():
        try:
            # Parse JSON arrays from CSV strings
            outcomes = json.loads(row.get('outcomes', '[]'))
            prices = json.loads(row.get('outcomePrices', '[]'))
            
            # Simple validation
            if len(outcomes) != len(prices) or len(outcomes) == 0:
                continue
                
            conditions = []
            
            # For simplicity in this POC, we treat each market row as a single Market object.
            # If it's a binary question (Yes/No), we create BOTH a "YES" and "NO" Condition 
            # so the LLM verifier logic works correctly.
            if len(outcomes) == 2 and set(o.lower() for o in outcomes) == {'yes', 'no'}:
                yes_idx = [i for i, o in enumerate(outcomes) if o.lower() == 'yes'][0]
                no_idx = 1 - yes_idx
                
                conditions.append(Condition(
                    condition_id=str(row['conditionId']) + "_yes",
                    question=f"Yes, {row['question']}",
                    description=str(row['description']),
                    token_id_yes=f"t_yes_{row['id']}",
                    token_id_no=f"t_no_{row['id']}",
                    outcome="Yes",
                    price_yes=float(prices[yes_idx]),
                    price_no=float(prices[no_idx])
                ))
                conditions.append(Condition(
                    condition_id=str(row['conditionId']) + "_no",
                    question=f"No, {row['question']}",
                    description=str(row['description']),
                    token_id_yes=f"t_no_{row['id']}",
                    token_id_no=f"t_yes_{row['id']}",
                    outcome="No",
                    price_yes=float(prices[no_idx]),
                    price_no=float(prices[yes_idx])
                ))
            else:
                # Mutually exclusive outcomes (Categorical)
                # We map each outcome to a condition to test combinatorial logic
                for idx, outcome in enumerate(outcomes):
                    conditions.append(Condition(
                        condition_id=f"{row['conditionId']}_{idx}",
                        question=f"Will {outcome} win: {row['question']}?",
                        description=str(row['description']),
                        token_id_yes=f"t_yes_{row['id']}_{idx}",
                        token_id_no=f"t_no_{row['id']}_{idx}",
                        outcome=outcome,
                        price_yes=float(prices[idx]),
                        price_no=1.0 - float(prices[idx]) # Approximation
                    ))
            
            # Parse End Date
            end_date = None
            if pd.notna(row.get('endDateIso')):
                try:
                    end_date = datetime.fromisoformat(str(row['endDateIso']).replace('Z', '+00:00'))
                except ValueError:
                    pass
            
            market = Market(
                market_id=str(row['id']),
                slug=str(row['slug']),
                question=str(row['question']),
                description=str(row['description']),
                conditions=conditions,
                end_date=end_date,
                topic=str(row['category']).capitalize() if pd.notna(row['category']) else "Unknown",
                tags=json.loads(row.get('tags', '[]')) if pd.notna(row.get('tags')) else [],
                is_negrisk=row.get('negRisk', False),
                total_liquidity=float(row.get('liquidity', 0.0)),
                total_volume=float(row.get('volume', 0.0))
            )
            markets.append(market)
            
        except Exception as e:
            # Skip rows that fail to parse
            continue
            
    return markets
