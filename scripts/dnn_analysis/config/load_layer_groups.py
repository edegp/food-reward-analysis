"""Load layer groups configuration from JSON file"""
import json
from pathlib import Path
from typing import Dict, List


def load_layer_groups() -> Dict[str, Dict[str, List[str]]]:
    """
    Load layer groups configuration from JSON file.

    Returns:
        Dictionary with layer group definitions for each source (convnext, clip)
        Format: {source: {group_label: [layer_names]}}
    """
    # Get the directory where this file is located
    config_dir = Path(__file__).parent
    config_file = config_dir / 'layer_groups.json'

    # Check if file exists
    if not config_file.exists():
        raise FileNotFoundError(f'Configuration file not found: {config_file}')

    # Load JSON
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Convert to simplified format for visualization scripts
    # Extract just the groups, converting from list format to dict format
    result = {}
    for source in ['convnext', 'clip']:
        result[source] = {}
        for group in config[source]['groups']:
            result[source][group['label']] = group['layers']

    return result


# Group order for sorting (shallow to deep)
GROUP_ORDER = ['Initial', 'Middle', 'Late', 'Final']
