#!/usr/bin/env python3
"""
Generate frame-by-frame text annotation files for IKEA Assembly videos.
Each line in the output txt file corresponds to the label for that frame number.
"""

import json
import os
import cv2
from pathlib import Path

def get_video_info(video_path):
    """Get the number of frames in a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def generate_frame_labels(annotations, total_frames):
    """Generate frame-by-frame labels from segment annotations."""
    frame_labels = ['NA'] * total_frames  # Default to 'NA'
    
    for annotation in annotations:
        label = annotation['label']
        start_frame, end_frame = annotation['segment']
        
        # Ensure we don't go beyond video length
        end_frame = min(end_frame, total_frames - 1)
        
        # Assign label to all frames in the segment
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx < total_frames:
                frame_labels[frame_idx] = label
    
    return frame_labels

def find_all_videos():
    """Find all video files in the dataset."""
    furniture_types = ['Kallax_Shelf_Drawer', 'Lack_Coffee_Table', 'Lack_Side_Table', 'Lack_TV_Bench']
    videos = []
    
    for furniture_type in furniture_types:
        furniture_path = Path(furniture_type)
        if not furniture_path.exists():
            print(f"Warning: {furniture_type} directory not found")
            continue
            
        for split in ['train', 'test']:
            split_path = furniture_path / split
            if not split_path.exists():
                print(f"Warning: {split_path} directory not found")
                continue
                
            # Find all subdirectories (video folders)
            for video_dir in split_path.iterdir():
                if video_dir.is_dir() and not video_dir.name.startswith('.'):
                    video_file = video_dir / 'dev3' / 'images' / 'scan_video.avi'
                    if video_file.exists():
                        # Create the annotation key that matches the JSON format
                        annotation_key = f"{furniture_type}/{video_dir.name}"
                        videos.append({
                            'video_path': video_file,
                            'annotation_key': annotation_key,
                            'output_dir': video_dir
                        })
                    else:
                        print(f"Warning: Video file not found at {video_file}")
    
    return videos

def main():
    # Load annotation file
    annotation_file = Path('annotations/action_annotations/gt_segments.json')
    if not annotation_file.exists():
        print(f"Error: Annotation file {annotation_file} not found")
        return
    
    with open(annotation_file, 'r') as f:
        annotation_data = json.load(f)
    
    # Find all videos
    videos = find_all_videos()
    print(f"Found {len(videos)} videos to process")
    
    processed_count = 0
    skipped_count = 0
    
    for video_info in videos:
        video_path = video_info['video_path']
        annotation_key = video_info['annotation_key']
        output_dir = video_info['output_dir']
        
        print(f"Processing: {annotation_key}")
        
        # Check if annotation exists
        if annotation_key not in annotation_data['database']:
            print(f"  Warning: No annotation found for {annotation_key}")
            skipped_count += 1
            continue
        
        # Get video frame count
        frame_count = get_video_info(video_path)
        if frame_count == 0:
            print(f"  Error: Could not read video {video_path}")
            skipped_count += 1
            continue
        
        print(f"  Video has {frame_count} frames")
        
        # Get annotations for this video
        video_annotations = annotation_data['database'][annotation_key]['annotation']
        
        # Generate frame-by-frame labels
        frame_labels = generate_frame_labels(video_annotations, frame_count)
        
        # Write to text file
        output_file = output_dir / 'frame_annotations.txt'
        with open(output_file, 'w') as f:
            for label in frame_labels:
                f.write(f"{label}\n")
        
        print(f"  Generated annotation file: {output_file}")
        processed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {processed_count} videos")
    print(f"Skipped: {skipped_count} videos")
    
    # Print summary of unique labels found
    all_labels = set()
    for video_data in annotation_data['database'].values():
        for annotation in video_data['annotation']:
            all_labels.add(annotation['label'])
    
    print(f"\nUnique labels found in dataset: {sorted(all_labels)}")

if __name__ == "__main__":
    main() 