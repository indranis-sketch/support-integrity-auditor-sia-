import json
from src.pseudo_label import rule_based_severity, resolution_time_severity

def generate_evidence_dossier(ticket_series):
    """
    Generates a structured JSON Evidence Dossier for a flagged ticket.
    """
    # Map priority to numeric for delta calculation
    priority_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    
    assigned_num = priority_map.get(ticket_series.get('Ticket Priority', 'Low'), 1)
    inferred_num = priority_map.get(ticket_series.get('inferred_severity', 'Low'), 1)
    
    delta = inferred_num - assigned_num
    
    if delta > 0:
        mismatch_type = "Hidden Crisis"
    elif delta < 0:
        mismatch_type = "False Alarm"
    else:
        mismatch_type = "Consistent" # Shouldn't happen if properly filtered
        
    text = str(ticket_series.get('Ticket Description', ''))
    
    evidence = []
    rule_score = rule_based_severity(text)
    if rule_score > 1:
        evidence.append({
            "signal": "keyword",
            "value": text[:50] + "..." if len(text) > 50 else text,
            "weight": f"{rule_score}/4"
        })
        
    time_score = resolution_time_severity(ticket_series.get('Resolution Time', 24))
    if time_score > 1:
        evidence.append({
            "signal": "resolution_time",
            "value": str(ticket_series.get('Resolution Time')),
            "interpretation": f"Resolved quickly indicating high severity (score {time_score}/4)"
        })
        
    dossier = {
        "ticket_id": str(ticket_series.get('Ticket ID', 'unknown')),
        "assigned_priority": str(ticket_series.get('Ticket Priority', 'Low')),
        "inferred_severity": str(ticket_series.get('inferred_severity', 'Low')),
        "mismatch_type": mismatch_type,
        "severity_delta": f"{delta:+} level(s)",
        "feature_evidence": evidence,
        "constraint_analysis": f"The ticket was assigned {ticket_series.get('Ticket Priority')} but inferred as {ticket_series.get('inferred_severity')}. Evidence from text keywords and resolution time suggests a mismatch.",
        "confidence": "High"
    }
    
    return dossier
