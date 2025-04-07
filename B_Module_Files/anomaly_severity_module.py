#!/usr/bin/env python
"""
Module: anomaly_severity_module.py

Purpose:
    This module provides a function to convert a continuous anomaly score into a categorical severity label.
    The severity levels are defined based on two thresholds:
      - "low" if the score is below the low_threshold,
      - "medium" if the score is between low_threshold and high_threshold,
      - "high" if the score is equal to or above high_threshold.
      
Usage:
    from B_Module_Files.anomaly_severity_module import get_anomaly_severity

    # Example:
    score = 0.7
    severity = get_anomaly_severity(score, low_threshold=0.5, high_threshold=0.8)
    print(f"Anomaly severity: {severity}")
"""

def get_anomaly_severity(score, low_threshold=0.5, high_threshold=0.8):
    """
    Converts a continuous anomaly score into a categorical severity label.

    Args:
        score (float): The continuous anomaly score output by the anomaly detection model.
        low_threshold (float): The lower threshold; scores below this are considered "low" severity.
        high_threshold (float): The higher threshold; scores equal to or above this are "high" severity.

    Returns:
        str: The anomaly severity category:
            - "low" if score < low_threshold,
            - "medium" if low_threshold <= score < high_threshold,
            - "high" if score >= high_threshold.
    """
    if score < low_threshold:
        return "low"
    elif score < high_threshold:
        return "medium"
    else:
        return "high"

if __name__ == "__main__":
    # Quick test of the module
    test_scores = [0.3, 0.55, 0.75, 0.9]
    for s in test_scores:
        severity = get_anomaly_severity(s, low_threshold=0.5, high_threshold=0.8)
        print(f"Score: {s} -> Severity: {severity}")