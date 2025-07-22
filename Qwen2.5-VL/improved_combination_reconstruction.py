import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re
from typing import List, Dict, Tuple, Optional

# Ground truth primitive tasks
PRIMITIVE_TASKS = ["insert the ball into the cylinder base", 
"insert the ball seat into the cylinder base", 
"insert the cylinder cap into the cylinder bracket", 
"insert the cylinder bracket into the cylinder base", 
"insert the large gear into the gear shaft", 
"insert the small gear into the gear shaft", 
"insert the bar into the hole for the bar", 
"insert the rod into the hole for the rod", 
"insert the large placer into the gear shaft", 
"insert the small placer into the gear shaft", 
"insert the screw bolt into the hole for the bolt", 
"insert the hex screw into the screw hole C1", 
"insert the hex screw into the screw hole C2", 
"insert the hex screw into the screw hole C3", 
"insert the hex screw into the screw hole C4", 
"insert the hex screw into the cyliner bracket", 
"insert the Phillips screw into the worm gear", 
"insert the usb male into the usb female", 
"insert the cylinder base into the ball seat", 
"insert the cylinder base into the cylinder bracket", 
"insert the cylinder cap into the cylinder base", 
"insert the cylinder bracket into the cylinder cap", 
"insert the gear shaft into the large gear", 
"insert the Phillips screw into the hole for the worm gear", 
"insert the Phillips screw into the hole for the Phillips screw", 
"slide the cylinder bracket", "slide the linear bearing", 
"place the cylinder base onto the assembly box", 
"place the cylinder bracket onto the assembly box", 
"place the worm gear onto the assembly box", 
"place the ball onto the ball seat", 
"place the ball seat onto the ball", 
"place the ball seat onto the assembly box", 
"place the ball seat on to the cylinder cap", 
"place the assembly box onto the desk", 
"place the cylinder cap onto the desk", 
"place the cylinder bracket onto the desk", 
"place the cylinder bracket onto the cylinder base", 
"place the cyinder subassembly onto the box", 
"place the large placer onto the large gear", 
"rotate the worm gear", 
"rotate the hand dial", 
"rotate the quarter-turn handle", 
"rotate the hand wheel", 
"screw the cylinder cap onto the cylinder base", 
"screw the gear shaft onto the hole for large gear", 
"screw the gear shaft onto the hole for large gear using the shaft wrench", 
"screw the gear shaft onto the hole for small gear", 
"screw the gear shaft onto the hole for small gear using the shaft wrench", 
"screw the nut onto the gear shaft", "screw the nut onto the gear shaft using the nut wrench", 
"screw the nut onto the stud on the assembly box", 
"screw the nut onto the stud on the assembly box using the nut wrench", 
"screw the nut onto the screw bolt", 
"screw the nut onto the screw bolt using the nut wrench", 
"screw the screw bolt onto the nut", 
"screw the hex screw into the screw hole C1", 
"screw the hex screw into the screw hole C1 using the hex screwdriver", 
"screw the hex screw into the screw hole C1 uing a Phillips screwdriver", 
"screw the hex screw into the screw hole C2", 
"screw the hex screw into the screw hole C2 using the hex screwdriver", 
"screw the hex screw into the screw hole C2 using the Phillips screwdriver", 
"screw the hex screw into the screw hole C3", 
"screw the hex screw into the screw hole C3 using the hex screwdriver", 
"screw the hex screw into the screw hole C3 using the Phillips screwdriver", 
"screw the hex screw into the screw hole C4", 
"screw the hex screw into the screw hole C4 using the hex screwdriver", 
"screw the hex screw into the screw hole C4 using the Phillips screwdriver", 
"screw the Phillips screw into the hole for worm gear", 
"screw the Phillips screw into the hole for worm gear using the Phillips screwdriver", 
"screw the Phillips screw into the hole for Phillips screw", 
"screw the Phillips screw into the hole for Phillips screw using the Phillips screwdriver", 
"screw the cylinder base into the cylinder cap", 
"null"]


possible_element_combinations = [
    ("insert", "ball", "cylinder base", "null"),
    ("insert", "ball seat", "cylinder base", "null"),
    ("insert", "cylinder cap", "cylinder bracket", "null"),
    ("insert", "cylinder bracket", "cylinder base", "null"),
    ("insert", "large gear", "gear shaft", "null"),
    ("insert", "small gear", "gear shaft", "null"),
    ("insert", "bar", "hole for the bar", "null"),
    ("insert", "rod", "hole for the rod", "null"),
    ("insert", "large placer", "gear shaft", "null"),
    ("insert", "small placer", "gear shaft", "null"),
    ("insert", "screw bolt", "hole for the bolt", "null"),
    ("insert", "hex screw", "screw hole c1", "null"),
    ("insert", "hex screw", "screw hole c2", "null"),
    ("insert", "hex screw", "screw hole c3", "null"),
    ("insert", "hex screw", "screw hole c4", "null"),
    ("insert", "hex screw", "cyliner bracket", "null"),
    ("insert", "phillips screw", "worm gear", "null"),
    ("insert", "usb male", "usb female", "null"),
    ("insert", "cylinder base", "ball seat", "null"),
    ("insert", "cylinder base", "cylinder bracket", "null"),
    ("insert", "cylinder cap", "cylinder base", "null"),
    ("insert", "cylinder bracket", "cylinder cap", "null"),
    ("insert", "gear shaft", "large gear", "null"),
    ("insert", "phillips screw", "hole for the worm gear", "null"),
    ("insert", "phillips screw", "hole for the phillips screw", "null"),
    ("slide", "cylinder bracket", "null", "null"),
    ("slide", "linear bearing", "null", "null"),
    ("place", "cylinder base", "assembly box", "null"),
    ("place", "cylinder bracket", "assembly box", "null"),
    ("place", "worm gear", "assembly box", "null"),
    ("place", "ball", "ball seat", "null"),
    ("place", "ball seat", "ball", "null"),
    ("place", "ball seat", "assembly box", "null"),
    ("place", "ball seat", "cylinder cap", "null"),
    ("place", "assembly box", "desk", "null"),
    ("place", "cylinder cap", "desk", "null"),
    ("place", "cylinder bracket", "desk", "null"),
    ("place", "cylinder bracket", "cylinder base", "null"),
    ("place", "cyinder subassembly", "box", "null"),
    ("place", "large placer", "large gear", "null"),
    ("rotate", "worm gear", "null", "null"),
    ("rotate", "hand dial", "null", "null"),
    ("rotate", "quarter-turn handle", "null", "null"),
    ("rotate", "hand wheel", "null", "null"),
    ("screw", "cylinder cap", "cylinder base", "null"),
    ("screw", "gear shaft", "hole for large gear", "null"),
    ("screw", "gear shaft", "hole for large gear", "shaft wrench"),
    ("screw", "gear shaft", "hole for small gear", "null"),
    ("screw", "gear shaft", "hole for small gear", "shaft wrench"),
    ("screw", "nut", "gear shaft", "null"),
    ("screw", "nut", "gear shaft", "nut wrench"),
    ("screw", "nut", "stud on the assembly box", "null"),
    ("screw", "nut", "stud on the assembly box", "nut wrench"),
    ("screw", "nut", "screw bolt", "null"),
    ("screw", "nut", "screw bolt", "nut wrench"),
    ("screw", "screw bolt", "nut", "null"),
    ("screw", "hex screw", "screw hole c1", "null"),
    ("screw", "hex screw", "screw hole c1", "hex screwdriver"),
    ("screw", "hex screw", "screw hole c1", "phillips screwdriver"),
    ("screw", "hex screw", "screw hole c2", "null"),
    ("screw", "hex screw", "screw hole c2", "hex screwdriver"),
    ("screw", "hex screw", "screw hole c2", "phillips screwdriver"),
    ("screw", "hex screw", "screw hole c3", "null"),
    ("screw", "hex screw", "screw hole c3", "hex screwdriver"),
    ("screw", "hex screw", "screw hole c3", "phillips screwdriver"),
    ("screw", "hex screw", "screw hole c4", "null"),
    ("screw", "hex screw", "screw hole c4", "hex screwdriver"),
    ("screw", "hex screw", "screw hole c4", "phillips screwdriver"),
    ("screw", "phillips screw", "hole for the worm gear", "null"),
    ("screw", "phillips screw", "hole for the worm gear", "phillips screwdriver"),
    ("screw", "phillips screw", "hole for the phillips screw", "null"),
    ("screw", "phillips screw", "hole for the phillips screw", "phillips screwdriver"),
    ("screw", "cylinder base", "cylinder cap", "null"),
    ("null", "null", "null", "null")
]

def parse_ground_truth_task(task: str) -> Tuple[str, str, str, str]:
    """
    Parse a ground truth primitive task string into (action_verb, manipulated_object, target_object, tool).
    """
    if not isinstance(task, str) or task.lower() == "null":
        return ("null", "null", "null", "null")
    
    # Regex to match: action the manipulated_object (into|onto)? the target_object (using the tool)?
    # Example: "insert the large placer into the gear shaft using the shaft wrench"
    pattern = re.compile(
        r"^(?P<action>\w+)\s+the\s+(?P<object>.+?)"
        r"(?:\s+(?P<prep>into|onto)\s+the\s+(?P<target>.+?))?"
        r"(?:\s+using\s+(?:the|a)\s+(?P<tool>.+))?$",
        re.IGNORECASE
    )
    match = pattern.match(task.strip())
    if not match:
        # Fallback: just return nulls if parsing fails
        return ("null", "null", "null", "null")
    
    action = match.group("action") or "null"
    obj = match.group("object") or "null"
    target = match.group("target") or "null"
    tool = match.group("tool") or "null"
    
    # If no preposition, target is null
    if not match.group("prep"):
        target = "null"
    
    return (action.lower(), obj.lower(), target.lower(), tool.lower())

def similarity(a, b):
    """Calculate similarity between two strings"""
    if pd.isna(a) or pd.isna(b) or str(a).lower() == 'null' or str(b).lower() == 'null':
        return 0.0
    return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

def clean_value(val):
    """Clean and standardize values"""
    if pd.isna(val) or str(val).lower() == 'null' or str(val).lower() == 'nan':
        return 'null'
    return str(val).strip().lower()

def correct_elements(action_verb: str, manipulated_object: str, target_object: str, tool: str) -> Tuple[str, str, str, str]:
    """Correct the elements based on the possible element combinations"""
    combination = (action_verb.lower(), manipulated_object.lower(), target_object.lower(), tool.lower())
    if combination in possible_element_combinations:
        return combination
    else:
        for possible in possible_element_combinations:
            possible_combination = (possible[0].lower(), possible[1].lower(), possible[2].lower(), possible[3].lower())
            if action_verb.lower() == possible[0].lower() and manipulated_object.lower() == possible[1].lower() and target_object.lower() == possible[2].lower():
                return possible_combination
            elif action_verb.lower() == possible[0].lower() and manipulated_object.lower() == possible[1].lower() and tool.lower() == possible[3].lower():
                return possible_combination
            elif action_verb.lower() == possible[0].lower() and target_object.lower() == possible[2].lower() and tool.lower() == possible[3].lower():
                return possible_combination
            elif manipulated_object.lower() == possible[1].lower() and target_object.lower() == possible[2].lower() and tool.lower() == possible[3].lower():
                return possible_combination
        return ("null", "null", "null", "null")

def combine_action_elements(action_verb: str, manipulated_object: str, target_object: str, tool: str) -> str:
    """Combine the four action elements to form a primitive task string"""
    
    # Clean and validate inputs
    action_verb = clean_value(action_verb)
    manipulated_object = clean_value(manipulated_object)
    target_object = clean_value(target_object)
    tool = clean_value(tool)
    corrected_combination = correct_elements(action_verb, manipulated_object, target_object, tool)

    return corrected_combination

def improved_combination_reconstruction(row: pd.Series) -> str:
    """Main function for improved combination reconstruction"""
    
    # Extract components
    action_verb = clean_value(row['action_verb'])
    manipulated_object = clean_value(row['manipulated_object'])
    target_object = clean_value(row['target_object'])
    tool = clean_value(row['tool'])
    
    # Step 1: Combine action elements with inference
    combined_task = combine_action_elements(action_verb, manipulated_object, target_object, tool)
    print(combined_task)
    input()
    
    return combined_task

def evaluate_accuracy(df: pd.DataFrame) -> Tuple[float, List[Dict]]:
    """Evaluate accuracy of the improved combination reconstruction"""
    correct_count = 0
    total_count = len(df)
    
    results = []
    
    for idx, row in df.iterrows():
        predicted_task_combination = improved_combination_reconstruction(row)
        ground_truth_task = row['primitive_task_gt']
        ground_truth_task_combination = parse_ground_truth_task(ground_truth_task)

        if predicted_task_combination == ground_truth_task_combination:
            is_correct = True
        else:
            is_correct = False

        if is_correct:
            correct_count += 1
        
        # Get combined task for analysis
        action_verb = clean_value(row['action_verb'])
        manipulated_object = clean_value(row['manipulated_object'])
        target_object = clean_value(row['target_object'])
        tool = clean_value(row['tool'])
        combined_task = combine_action_elements(action_verb, manipulated_object, target_object, tool)
        
        results.append({
            'filename': row['filename'],
            'original_task': row['primitive_task'],
            'ground_truth_task_combination': ground_truth_task_combination,
            'predicted_task_combination': predicted_task_combination,
            'ground_truth_task': ground_truth_task,
            'is_correct': is_correct,
        })
    
    accuracy = correct_count / total_count * 100
    
    return accuracy, results

def main():
    # Load the CSV data 
    root_dir = './lh_v0'
    df = pd.read_csv(f'{root_dir}/action_verb_objects_tool_19.csv')
    
    print("Original Primitive Task Accuracy:")
    original_correct = df['is_primitive_task_correct'].sum()
    original_total = len(df)
    original_accuracy = original_correct / original_total * 100
    print(f"Accuracy: {original_accuracy:.2f}% ({original_correct}/{original_total})")
    
    print("\nApplying Improved Combination Reconstruction...")
    new_accuracy, results = evaluate_accuracy(df)
    
    print(f"\nNew Primitive Task Accuracy: {new_accuracy:.2f}%")
    print(f"Improvement: {new_accuracy - original_accuracy:.2f} percentage points")
    print(f"Relative Improvement: {((new_accuracy - original_accuracy) / original_accuracy * 100):.1f}%")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{root_dir}/improved_combination_results.csv', index=False)
    
    # Show some examples
    print("\nExample reconstructions:")
    for i, result in enumerate(results[:10]):
        if result['original_task'] != result['predicted_task_combination']:
            print(f"File: {result['filename']}")
            print(f"  Original: {result['original_task']}")
            print(f"  ground_truth_task_combination: {result['ground_truth_task_combination']}")
            print(f"  Predicted: {result['predicted_task_combination']}")
            print(f"  Ground Truth: {result['ground_truth_task']}")
            print(f"  Correct: {result['is_correct']}")
            print()

if __name__ == "__main__":
    main() 