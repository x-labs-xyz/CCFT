#!/usr/bin/env python3
"""
Convert annotation files from old underscore format to new verb_object format
where multi-word verbs/objects use hyphens (-) internally and underscore (_) separates verb from object.
"""

import os
from pathlib import Path

def create_old_to_new_format_mapping():
    """Create mapping from old underscore format to new verb_object format."""
    mapping = {
        # Convert from old format to new format with clearer verb_object separation
        "NA": "NA",
        "align_leg": "align_leg",
        "align_side_panel": "align_side-panel", 
        "attach_back_panel": "attach_back-panel",
        "attach_side_panel": "attach_side-panel",
        "attach_shelf": "attach_shelf",
        "flip_shelf": "flip_shelf",
        "flip_table": "flip_table",
        "flip_table_top": "flip_table-top",
        "insert_pin": "insert_pin",
        "lay_down_back_panel": "lay-down_back-panel",
        "lay_down_bottom_panel": "lay-down_bottom-panel", 
        "lay_down_front_panel": "lay-down_front-panel",
        "lay_down_leg": "lay-down_leg",
        "lay_down_shelf": "lay-down_shelf",
        "lay_down_side_panel": "lay-down_side-panel",
        "lay_down_table_top": "lay-down_table-top",
        "other": "other",
        "pick_up_back_panel": "pick-up_back-panel",
        "pick_up_bottom_panel": "pick-up_bottom-panel",
        "pick_up_front_panel": "pick-up_front-panel", 
        "pick_up_leg": "pick-up_leg",
        "pick_up_pin": "pick-up_pin",
        "pick_up_shelf": "pick-up_shelf",
        "pick_up_side_panel": "pick-up_side-panel",
        "pick_up_table_top": "pick-up_table-top",
        "position_drawer": "position_drawer",
        "push_table": "push_table",
        "push_table_top": "push_table-top",
        "rotate_table": "rotate_table", 
        "slide_bottom_panel": "slide_bottom-panel",
        "spin_leg": "spin_leg",
        "tighten_leg": "tighten_leg"
    }
    return mapping

def convert_annotation_file(file_path, mapping):
    """Convert a single annotation file from old to new format."""
    print(f"Converting: {file_path.name}")
    
    # Read the original file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Convert each line
    converted_lines = []
    unmapped_labels = set()
    
    for line in lines:
        old_label = line.strip()
        if old_label in mapping:
            converted_lines.append(mapping[old_label] + '\n')
        else:
            # Keep original if not found in mapping
            converted_lines.append(line)
            if old_label:  # Don't add empty lines to unmapped
                unmapped_labels.add(old_label)
    
    # Write back to the same file
    with open(file_path, 'w') as f:
        f.writelines(converted_lines)
    
    if unmapped_labels:
        print(f"  Warning: Unmapped labels found: {unmapped_labels}")
    
    return len(unmapped_labels) == 0

def main():
    # Create the mapping from old to new format
    mapping = create_old_to_new_format_mapping()
    
    # Get target directory from command line or default to test
    import sys
    if len(sys.argv) > 1:
        target_dir = Path(f'IKEA_annotations/{sys.argv[1]}')
    else:
        target_dir = Path('IKEA_annotations/test')
    
    if not target_dir.exists():
        print(f"Error: Directory {target_dir} not found")
        return
    
    # Get all annotation files
    annotation_files = list(target_dir.glob('*.txt'))
    
    if not annotation_files:
        print(f"No .txt files found in {target_dir}")
        return
        
    print(f"Converting files in: {target_dir}")
    print(f"Found {len(annotation_files)} annotation files to convert")
    print(f"Using mapping with {len(mapping)} entries")
    print("Converting from old underscore format to new verb_object format")
    print("=" * 70)
    
    # Convert each file
    successful_conversions = 0
    for file_path in annotation_files:
        success = convert_annotation_file(file_path, mapping)
        if success:
            successful_conversions += 1
    
    print("=" * 70)
    print(f"Conversion complete!")
    print(f"Successfully converted: {successful_conversions}/{len(annotation_files)} files")
    
    # Show some examples of converted content
    print("\nSample of converted content (first 10 lines of first file):")
    if annotation_files:
        with open(annotation_files[0], 'r') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"  {line.strip()}")

if __name__ == "__main__":
    main() 