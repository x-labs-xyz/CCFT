#!/usr/bin/env python3
"""
Organize IKEA assembly videos by action labels and generate label mapping file.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

def extract_action_label(filename):
    """
    Extract the full action label from a video filename.
    Examples:
    - "0036_spin_leg_001.mp4" → "spin_leg"
    - "0031_pick-up_leg_001.mp4" → "pick-up_leg"
    - "0031_lay-down_table-top_001.mp4" → "lay-down_table-top"
    - "0037_align_side-panel_001.mp4" → "align_side-panel"
    - "0031_NA_001.mp4" → "NA"
    """
    # Remove extension and parse
    name_no_ext = filename.replace('.mp4', '')
    parts = name_no_ext.split('_')
    
    if len(parts) < 3:
        return None
    
    # Skip video ID (first part) and clip index (last part)
    # Extract the full action label (everything in between)
    video_id = parts[0]
    clip_index = parts[-1]
    
    # Everything between video_id and clip_index is the action label
    action_label_parts = parts[1:-1]
    action_label = '_'.join(action_label_parts)
    
    # Replace hyphens with underscores for folder names (to avoid filesystem issues)
    return action_label.replace('-', '_')

def organize_videos_by_labels(source_dir, target_base_dir):
    """
    Organize videos by action labels and return mapping information.
    """
    source_path = Path(source_dir)
    target_path = Path(target_base_dir)
    
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist")
        return [], {}
    
    # Collect all videos and their labels
    video_label_mapping = []
    label_counts = defaultdict(int)
    
    # Process all video files
    for video_file in source_path.glob('*.mp4'):
        action_label = extract_action_label(video_file.name)
        if action_label is None:
            print(f"Warning: Could not extract action label from {video_file.name}")
            continue
        
        video_label_mapping.append((video_file, action_label))
        label_counts[action_label] += 1
    
    print(f"Found {len(video_label_mapping)} videos with {len(label_counts)} unique action labels")
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Create target directories and move videos
    moved_videos = []
    for video_file, action_label in video_label_mapping:
        # Create target directory for this label
        label_dir = target_path / action_label
        label_dir.mkdir(parents=True, exist_ok=True)
        
        # Define target path
        target_video_path = label_dir / video_file.name
        
        # Copy (not move) the video to preserve original structure
        try:
            shutil.copy2(video_file, target_video_path)
            # Store relative path from target_base_dir
            relative_path = f"{action_label}/{video_file.name}"
            moved_videos.append((relative_path, action_label))
        except Exception as e:
            print(f"Error copying {video_file.name}: {e}")
    
    return moved_videos, label_counts

def create_label_mapping(video_paths_labels, output_file):
    """
    Create a mapping file with video paths and label numbers.
    """
    # Get unique labels and sort them to ensure consistent numbering
    unique_labels = sorted(set(label for _, label in video_paths_labels))
    
    # Create label to number mapping
    label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"\nLabel to number mapping:")
    for label, num in label_to_num.items():
        print(f"  {num}: {label}")
    
    # Write the mapping file
    with open(output_file, 'w') as f:
        for video_path, label in sorted(video_paths_labels):
            label_num = label_to_num[label]
            f.write(f"{video_path} {label_num}\n")
    
    return label_to_num

def main():
    base_dir = Path('IKEA_mmaction')
    
    # Process training videos
    print("=" * 60)
    print("Processing TRAINING videos")
    print("=" * 60)
    
    train_videos, train_label_counts = organize_videos_by_labels(
        base_dir / 'videos_train',
        base_dir / 'train'
    )
    
    # Process validation videos
    print("\n" + "=" * 60)
    print("Processing VALIDATION videos")
    print("=" * 60)
    
    val_videos, val_label_counts = organize_videos_by_labels(
        base_dir / 'videos_val', 
        base_dir / 'val'
    )
    
    # Combine all videos for unified label mapping
    all_videos = train_videos + val_videos
    
    # Create unified label mapping
    print("\n" + "=" * 60)
    print("Creating label mapping files")
    print("=" * 60)
    
    # Create combined mapping file
    label_mapping = create_label_mapping(all_videos, base_dir / 'video_label_mapping.txt')
    
    # Create separate train and val mapping files
    create_label_mapping(train_videos, base_dir / 'train_label_mapping.txt')
    create_label_mapping(val_videos, base_dir / 'val_label_mapping.txt')
    
    # Save label definitions
    with open(base_dir / 'label_definitions.txt', 'w') as f:
        f.write("# Label number to action label mapping\n")
        for num, label in enumerate(sorted(label_mapping.keys())):
            f.write(f"{num}: {label}\n")
    
    print(f"\nSUMMARY:")
    print(f"  Training videos: {len(train_videos)}")
    print(f"  Validation videos: {len(val_videos)}")
    print(f"  Total videos: {len(all_videos)}")
    print(f"  Unique action labels: {len(label_mapping)}")
    print(f"\nFiles created:")
    print(f"  {base_dir}/video_label_mapping.txt - Combined mapping")
    print(f"  {base_dir}/train_label_mapping.txt - Training set mapping")
    print(f"  {base_dir}/val_label_mapping.txt - Validation set mapping")
    print(f"  {base_dir}/label_definitions.txt - Label definitions")
    print(f"\nVideos organized in:")
    print(f"  {base_dir}/train/[action_label]/ - Training videos by label")
    print(f"  {base_dir}/val/[action_label]/ - Validation videos by label")

if __name__ == "__main__":
    main() 