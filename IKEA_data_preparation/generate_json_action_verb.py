#!/usr/bin/env python3
"""
Generate JSON files for action verb recognition from IKEA Assembly video clips.
"""

import os
import json
from pathlib import Path
from collections import defaultdict

def extract_action_verb(label):
    """
    Extract the action verb from a label.
    Examples:
    - "spin_leg" → "spin"
    - "pick-up_leg" → "pick up"
    - "align_leg" → "align"
    - "NA" → "NA"
    - "other" → "other"
    """
    if label in ["NA", "other"]:
        return label
    
    # Split by underscore and take the first part (verb)
    parts = label.split('_')
    if len(parts) >= 2:
        action_verb = parts[0]  # Get the verb part
        # Replace hyphens with spaces for natural language
        return action_verb.replace('-', ' ')
    else:
        # Replace hyphens with spaces for natural language
        return label.replace('-', ' ')

def parse_video_filename(filename):
    """
    Parse video filename to extract components.
    Format: {video_id}_{label}_{clip_index}.mp4
    Returns: (video_id, label, clip_index) or None if invalid
    """
    if not filename.lower().endswith('.mp4'):
        return None
    
    # Remove extension
    name_no_ext = filename[:-4]
    
    # Split by underscore
    parts = name_no_ext.split('_')
    
    if len(parts) < 3:
        return None
    
    # Extract components
    video_id = parts[0]
    clip_index = parts[-1]
    
    # Everything between video_id and clip_index is the label
    label_parts = parts[1:-1]
    label = '_'.join(label_parts)
    
    try:
        clip_index = int(clip_index)
    except ValueError:
        return None
    
    return (video_id, label, clip_index)

def create_annotation_entry(filename, label, action_verb, split_type):
    """
    Create a JSON annotation entry for action verb recognition.
    """
    user_text = "<video>What is the action verb performed in this IKEA assembly video?"
    
    if action_verb in ["NA", "other"]:
        assistant_text = f"The action verb is NA."
    else:
        assistant_text = f"The action verb is {action_verb}."
    
    return {
        "messages": [
            {
                "content": user_text,
                "role": "user"
            },
            {
                "content": assistant_text,
                "role": "assistant"
            }
        ],
        "videos": [
            f"IKEA_video_clips/{split_type}/{filename}"
        ]
    }

def process_split(split_type, video_clips_dir):
    """
    Process all videos in a split (train or test) and generate annotations.
    """
    split_dir = video_clips_dir / split_type
    if not split_dir.exists():
        print(f"Warning: {split_dir} does not exist")
        return []
    
    annotations = []
    action_verb_counts = defaultdict(int)
    
    # Process each video clip directly in the split directory
    for video_file in split_dir.glob('*.mp4'):
        parsed = parse_video_filename(video_file.name)
        if parsed is None:
            continue
            
        video_id, label, clip_index = parsed
        action_verb = extract_action_verb(label)
        
        # Create annotation entry with just the filename
        entry = create_annotation_entry(video_file.name, label, action_verb, split_type)
        annotations.append(entry)
        
        action_verb_counts[action_verb] += 1
    
    print(f"{split_type.upper()} set:")
    print(f"  Total clips: {len(annotations)}")
    print(f"  Unique action verbs: {len(action_verb_counts)}")
    print(f"  Action verb distribution:")
    for verb, count in sorted(action_verb_counts.items()):
        print(f"    {verb}: {count}")
    
    return annotations

def main():
    video_clips_dir = Path('IKEA_video_clips')
    
    if not video_clips_dir.exists():
        print(f"Error: {video_clips_dir} directory not found")
        return
    
    # Create output directory
    output_dir = Path('json_annotations')
    output_dir.mkdir(exist_ok=True)
    
    # Process both splits
    all_annotations = []
    
    for split_type in ['train', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split_type.upper()} split for action verb recognition")
        print(f"{'='*60}")
        
        split_annotations = process_split(split_type, video_clips_dir)
        all_annotations.extend(split_annotations)
        
        # Save split-specific file
        split_output = output_dir / f"ikea_assembly_action_verb_{split_type}.json"
        with open(split_output, 'w', encoding='utf-8') as f:
            json.dump(split_annotations, f, indent=2)
        print(f"  Saved {len(split_annotations)} annotations to {split_output}")
    
    # Save combined file
    combined_output = output_dir / "ikea_assembly_action_verb_combined.json"
    with open(combined_output, 'w', encoding='utf-8') as f:
        json.dump(all_annotations, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY - Action Verb Recognition")
    print(f"{'='*60}")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"Combined file: {combined_output}")
    print(f"Split files: {output_dir}/ikea_assembly_action_verb_{{train,test}}.json")

if __name__ == "__main__":
    main() 