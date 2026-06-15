import pandas as pd
import numpy as np

def rule_based_severity(text):
    """
    Rule-based NLP features: keyword density, escalation phrases.
    Returns a score from 1 (Low) to 4 (Critical).
    """
    text = str(text).lower()
    escalation_words = ['urgent', 'asap', 'immediate', 'critical', 'emergency', 'down', 'crash', 'losing']
    high_words = ['broken', 'error', 'fail', 'block', 'issue']
    
    score = 1
    if any(word in text for word in escalation_words):
        score = 4
    elif any(word in text for word in high_words):
        score = 3
    elif len(text.split()) > 50: # Longer complaints might be medium
        score = 2
        
    return score

def resolution_time_severity(time_hrs):
    """
    Resolution-time regression proxy.
    Assume Critical tickets are resolved in < 4 hours, High in < 12, Medium in < 24, Low > 24.
    Returns a score from 1 (Low) to 4 (Critical).
    """
    try:
        time_hrs = float(time_hrs)
    except:
        return 1
        
    if time_hrs <= 4: return 4
    if time_hrs <= 12: return 3
    if time_hrs <= 24: return 2
    return 1

def generate_pseudo_labels(df):
    """
    Fuses independent signals to infer severity and generates binary mismatch labels.
    """
    print("Generating pseudo-labels...")
    
    df['rule_score'] = df['Ticket Description'].apply(rule_based_severity)
    df['time_score'] = df['Resolution Time'].apply(resolution_time_severity)
    
    # Fusion strategy: Max of the two scores (conservative approach to not miss criticals)
    df['inferred_severity_score'] = df[['rule_score', 'time_score']].max(axis=1)
    
    # Map back to labels
    score_to_label = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Critical'}
    df['inferred_severity'] = df['inferred_severity_score'].map(score_to_label)
    
    # Binary mismatch label: 1 if human-assigned Priority != inferred_severity, else 0
    df['mismatch_label'] = (df['Ticket Priority'] != df['inferred_severity']).astype(int)
    
    print(f"Found {df['mismatch_label'].sum()} mismatched tickets out of {len(df)}.")
    return df
