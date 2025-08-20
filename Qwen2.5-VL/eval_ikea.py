import os
import re
import spacy
import numpy as np
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer, util
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import pandas as pd
import argparse  

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

ACTION_VERBS = ["insert", "slide", "place", "rotate", "screw", "wrong", "null"]                   
MANIPULATED_OBJS  = ["ball", "assembly box", "ball seat", "cylinder base", "cylinder cap", "cylinder bracket", "cylinder subassembly", "gear shaft", "large gear", "small gear", "bar", "rod", "large placer", "small placer", "screw bolt", "hex screw", "Phillips screw", "usb male", "linear bearing", "worm gear", "hand wheel", "quarter-turn handle", "hand dial", "nut", "screw bolt", "null"]                    
TARGET_OBJS  = ["ball", "assembly box", "cylinder base", "ball seat", "cylinder bracket", "cylinder cap", "large gear", "gear shaft", "hole for the rod", "hole for the bar", "hole for the bolt", "hole for the Phillips screw", "stud on the assembly box", "usb female", "screw hole C1", "screw hole C2", "screw hole C3", "screw hole C4", "worm gear", "hole for the large gear", "hole for the small gear", "hole for the worm gear", "screw bolt", "nut", "null"]                    
TOOLS = ["hex screwdriver", "Phillips screwdriver", "shaft wrench", "nut wrench", "null"]                    
PRIMITIVE_TASKS = ["insert the ball into the cylinder base", "insert the ball seat into the cylinder base", "insert the cylinder cap into the cylinder bracket", "insert the cylinder bracket into the cylinder base", "insert the large gear into the gear shaft", "insert the small gear into the gear shaft", "insert the bar into the hole for the bar", "insert the rod into the hole for the rod", "insert the large placer into the gear shaft", "insert the small placer into the gear shaft", "insert the screw bolt into the hole for the bolt", "insert the hex screw into the screw hole C1", "insert the hex screw into the screw hole C2", "insert the hex screw into the screw hole C3", "insert the hex screw into the screw hole C4", "insert the hex screw into the cyliner bracket", "insert the Phillips screw into the worm gear", "insert the usb male into the usb female", "insert the cylinder base into the ball seat", "insert the cylinder base into the cylinder bracket", "insert the cylinder cap into the cylinder base", "insert the cylinder bracket into the cylinder cap", "insert the gear shaft into the large gear", "insert the Phillips screw into the hole for the worm gear", "insert the Phillips screw into the hole for the Phillips screw", "slide the cylinder bracket", "slide the linear bearing", "place the cylinder base onto the assembly box", "place the cylinder bracket onto the assembly box", "place the worm gear onto the assembly box", "place the ball onto the ball seat", "place the ball seat onto the ball", "place the ball seat onto the assembly box", "place the ball seat on to the cylinder cap", "place the assembly box onto the desk", "place the cylinder cap onto the desk", "place the cylinder bracket onto the desk", "place the cylinder bracket onto the cylinder base", "place the cyinder subassembly onto the box", "place the large placer onto the large gear", "rotate the worm gear", "rotate the hand dial", "rotate the quarter-turn handle", "rotate the hand wheel", "screw the cylinder cap onto the cylinder base", "screw the gear shaft onto the hole for large gear", "screw the gear shaft onto the hole for large gear using the shaft wrench", "screw the gear shaft onto the hole for small gear", "screw the gear shaft onto the hole for small gear using the shaft wrench", "screw the nut onto the gear shaft", "screw the nut onto the gear shaft using the nut wrench", "screw the nut onto the stud on the assembly box", "screw the nut onto the stud on the assembly box using the nut wrench", "screw the nut onto the screw bolt", "screw the nut onto the screw bolt using the nut wrench", "screw the screw bolt onto the nut", "screw the hex screw into the screw hole C1", "screw the hex screw into the screw hole C1 using the hex screwdriver", "screw the hex screw into the screw hole C1 uing a Phillips screwdriver", "screw the hex screw into the screw hole C2", "screw the hex screw into the screw hole C2 using the hex screwdriver", "screw the hex screw into the screw hole C2 using the Phillips screwdriver", "screw the hex screw into the screw hole C3", "screw the hex screw into the screw hole C3 using the hex screwdriver", "screw the hex screw into the screw hole C3 using the Phillips screwdriver", "screw the hex screw into the screw hole C4", "screw the hex screw into the screw hole C4 using the hex screwdriver", "screw the hex screw into the screw hole C4 using the Phillips screwdriver", "screw the Phillips screw into the hole for worm gear", "screw the Phillips screw into the hole for worm gear using the Phillips screwdriver", "screw the Phillips screw into the hole for Phillips screw", "screw the Phillips screw into the hole for Phillips screw using the Phillips screwdriver", "screw the cylinder base into the cylinder cap", "null"]
PRIMITIVE_TASKS_no_null = ["insert the ball into the cylinder base", "insert the ball seat into the cylinder base", "insert the cylinder cap into the cylinder bracket", "insert the cylinder bracket into the cylinder base", "insert the large gear into the gear shaft", "insert the small gear into the gear shaft", "insert the bar into the hole for the bar", "insert the rod into the hole for the rod", "insert the large placer into the gear shaft", "insert the small placer into the gear shaft", "insert the screw bolt into the hole for the bolt", "insert the hex screw into the screw hole C1", "insert the hex screw into the screw hole C2", "insert the hex screw into the screw hole C3", "insert the hex screw into the screw hole C4", "insert the hex screw into the cyliner bracket", "insert the Phillips screw into the worm gear", "insert the usb male into the usb female", "insert the cylinder base into the ball seat", "insert the cylinder base into the cylinder bracket", "insert the cylinder cap into the cylinder base", "insert the cylinder bracket into the cylinder cap", "insert the gear shaft into the large gear", "insert the Phillips screw into the hole for the worm gear", "insert the Phillips screw into the hole for the Phillips screw", "slide the cylinder bracket", "slide the linear bearing", "place the cylinder base onto the assembly box", "place the cylinder bracket onto the assembly box", "place the worm gear onto the assembly box", "place the ball onto the ball seat", "place the ball seat onto the ball", "place the ball seat onto the assembly box", "place the ball seat on to the cylinder cap", "place the assembly box onto the desk", "place the cylinder cap onto the desk", "place the cylinder bracket onto the desk", "place the cylinder bracket onto the cylinder base", "place the cyinder subassembly onto the box", "place the large placer onto the large gear", "rotate the worm gear", "rotate the hand dial", "rotate the quarter-turn handle", "rotate the hand wheel", "screw the cylinder cap onto the cylinder base", "screw the gear shaft onto the hole for large gear", "screw the gear shaft onto the hole for large gear using the shaft wrench", "screw the gear shaft onto the hole for small gear", "screw the gear shaft onto the hole for small gear using the shaft wrench", "screw the nut onto the gear shaft", "screw the nut onto the gear shaft using the nut wrench", "screw the nut onto the stud on the assembly box", "screw the nut onto the stud on the assembly box using the nut wrench", "screw the nut onto the screw bolt", "screw the nut onto the screw bolt using the nut wrench", "screw the screw bolt onto the nut", "screw the hex screw into the screw hole C1", "screw the hex screw into the screw hole C1 using the hex screwdriver", "screw the hex screw into the screw hole C1 uing a Phillips screwdriver", "screw the hex screw into the screw hole C2", "screw the hex screw into the screw hole C2 using the hex screwdriver", "screw the hex screw into the screw hole C2 using the Phillips screwdriver", "screw the hex screw into the screw hole C3", "screw the hex screw into the screw hole C3 using the hex screwdriver", "screw the hex screw into the screw hole C3 using the Phillips screwdriver", "screw the hex screw into the screw hole C4", "screw the hex screw into the screw hole C4 using the hex screwdriver", "screw the hex screw into the screw hole C4 using the Phillips screwdriver", "screw the Phillips screw into the hole for worm gear", "screw the Phillips screw into the hole for worm gear using the Phillips screwdriver", "screw the Phillips screw into the hole for Phillips screw", "screw the Phillips screw into the hole for Phillips screw using the Phillips screwdriver", "screw the cylinder base into the cylinder cap"]



def load_object_mapping(file_path):
    """
    Reads a file where each line has the format:
        XX "semantic name"
    For example:
        ba "ball"
        bs "ball seat"

    Returns a dictionary mapping:
        {"ba": "ball", "bs": "ball seat", ...}
    """
    mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split into code and name (strip quotes from name)
            code, name = line.split(" ", 1)
            name = name.strip().strip('"')
            mapping[code] = name
    return mapping

def load_all_mappings(base_dir):
    """Load all mapping files from a directory and return them as a tuple.
    
    Args:
        base_dir (str): Path to directory containing mapping files. Defaults to "./eval_utils/"
        
    Returns:
        tuple: Contains four mappings in order:
            - action_verb_mapping
            - object_mapping
            - tool_mapping
            - label_mapping
    """
    def load(path):
        return load_object_mapping(os.path.join(base_dir, path))
    
    return (
        load("action_verb_mapping.txt"),
        load("object_mapping.txt"),
        load("tool_mapping.txt"),
        load("label_mapping.txt"),
    )

def map_label_with_semantics(label_element, mapping_file):
    # Handle special cases
    if label_element == "null":
        return "null"
    elif label_element == "wrong":
        return "wrong"

    # Map codes to semantic names (if a code isn't found, fallback to the raw code)
    semantics = mapping_file.get(label_element, label_element)

    return semantics

def parse_label(label):
    """
    Splits a label into four elements according to these rules:
      - If the label == "null": all four elements are None.
      - If the label == "wrong": all four elements are "wrong".
      - Otherwise:
          * action_verb         = label[0]
          * manipulated_object  = label[1:3]
          * target_object       = label[3:5]
          * tool               = label[5:7]

    Returns a tuple: (action_verb, manipulated_object, target_object, tool).
    """
    # Special cases
    if label == "null":
        return ("null", "null", "null", "null")
    elif label == "wrong":
        return ("wrong", "wrong", "wrong", "wrong")

    # General parsing
    action_verb         = label[0]     if len(label) >= 1 else "null"
    manipulated_object  = label[1:3]   if len(label) >= 3 else "null"
    target_object       = label[3:5]   if len(label) >= 5 else "null"
    tool                = label[5:7]   if len(label) >= 7 else "null"

    return (action_verb, manipulated_object, target_object, tool)

# Extract Ground Truth Label from Filename
def extract_label(filename):
    # Split filename by underscores and extract the label part (e.g., "ibacb" in "S04A04I01M0_ibacb_2.mp4")
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {filename}")
    return parts[1]  # Assumes format: name_label_index.mp4

def exact_match(response, task_list):
    for task in task_list:
        if task in response:
            return task
    return None  # No exact match found

# Parse VLM Output to Extract Primitive Task
def parse_vlm_output(vlm_text):
    # Use regex to find the task description (e.g., "screw the nut onto the gear shaft")
    patterns = {
        "action_verb": r"action verb is\s*['\"]?([\w-]+(?:\s+[\w-]+)*)",
        "object": r"object is\s*['\"]?([\w-]+(?:\s+[\w-]+)*)",
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, vlm_text, re.IGNORECASE)
        extracted[key] = match.group(1) if match else "NA"

    if extracted["action_verb"] == "NA":
        reconstructed_primitive_task = "NA"
    elif extracted["object"] == "NA":
        reconstructed_primitive_task = "NA"
    else:
        reconstructed_primitive_task = f"{extracted['action_verb']} {extracted['object']}"
    
    extracted["primitive_task"] = reconstructed_primitive_task
    return extracted

# Compute Semantic Similarity
def semantic_similarity(text1, text2):
    embeddings = sbert_model.encode([text1, text2])
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0.0

def get_video_label(video_path):
    """
    Extract the action verb, object, and whole label from a video path.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: (action_verb, object, whole_label) - with spaces for readability
        
    Examples:
        >>> get_video_label("path/to/0036_spin_leg_001.mp4")
        ('spin', 'leg', 'spin leg')
        >>> get_video_label("0031_pick-up_leg_001.mp4")
        ('pick up', 'leg', 'pick up leg')
        >>> get_video_label("train/align_side_panel/0037_align_side-panel_001.mp4")
        ('align', 'side panel', 'align side panel')
        >>> get_video_label("0031_NA_001.mp4")
        ('NA', '', 'NA')
    """
    # Get just the filename from the path
    filename = os.path.basename(video_path)
    
    # Remove extension
    name_no_ext = filename.replace('.mp4', '')
    
    # Split by underscore
    parts = name_no_ext.split('_')
    
    if len(parts) < 3:
        return None, None, None
    
    # Skip video ID (first part) and clip index (last part)
    # Extract the full action label (everything in between)
    action_label_parts = parts[1:-1]
    action_label = '_'.join(action_label_parts)
    
    # Handle special cases like NA or other
    if action_label in ['NA', 'other']:
        return action_label, '', action_label
    
    # Split action label into verb and object parts
    # First part is verb, rest is object
    if '_' in action_label:
        label_parts = action_label.split('_')
        verb_part = label_parts[0]
        object_parts = label_parts[1:]
        object_part = '_'.join(object_parts)
    else:
        # Single word case
        verb_part = action_label
        object_part = 'NA'
    
    # Clean each part by replacing hyphens and underscores with spaces
    clean_verb = verb_part.replace('-', ' ')
    clean_object = object_part.replace('_', ' ').replace('-', ' ')
    
    # Create whole label
    if clean_object:
        clean_whole = f"{clean_verb} {clean_object}"
    else:
        clean_whole = clean_verb
    
    return clean_verb, clean_object, clean_whole

# Main evaluation workflow
def evaluate_vlm(video_folder, processor, llm, log_file_path, wrong_preds_path, txt_file_path, similarity_threshold=0.9):
    # Initialize counters and results storage
    total = 0
    correct_primitive_task = 0
    correct_action_verb = 0
    correct_object = 0
    results_log = []

    # Iterate through video files
    for filename in os.listdir(video_folder):
        if not filename.endswith(".mp4"):
            continue

        entry = {
            "filename": filename,
            "gt_label": None,
            "vlm_output_action_verb": None,
            "vlm_output_objects": None,
            "vlm_output": None,
            "primitive_task": None,
            "primitive_task_gt": None,
            "action_verb": None,
            "action_verb_gt": None,
            "object": None,
            "object_gt": None,
            "primitive_task_similarity": None,
            "action_verb_similarity": None,
            "object_similarity": None,
            "is_primitive_task_correct": False,
            "is_action_verb_correct": False,
            "is_object_correct": False,
        }

        try:
            # Extract ground truth
            action_verb_gt, object_gt, primitive_task_gt = get_video_label(filename)
            entry["primitive_task_gt"] = primitive_task_gt
            
            # VLM inference
            video_path = os.path.join(video_folder, filename)
            messages_action_verb = [{"role": "system", "content": " "},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "What is the action verb performed in this IKEA assembly video?"},
                                        {"type": "video", 
                                        "video": f"{video_path}",   #/workspace/LLaMA-Factory/data/CCFT_lh_de_513/S04A04I01M0_sshc1dh_16.mp4  /workspace/MiniCPM-o/assets/CCFT/S04A04I01M0_ibacb_2.mp4
                                        "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28}
                                        ]
                                }]
            prompt_action_verb = processor.apply_chat_template( messages_action_verb,
                                                    tokenize=False,
                                                    add_generation_prompt=True,)

            image_inputs, video_inputs, video_kwargs = process_vision_info(messages_action_verb, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            llm_inputs_action_verb = {"prompt": prompt_action_verb,
                        "multi_modal_data": mm_data,
                        # FPS will be returned in video_kwargs
                        "mm_processor_kwargs": video_kwargs,}

            outputs_action_verb = llm.generate([llm_inputs_action_verb], sampling_params=sampling_params)

            vlm_output_action_verb = outputs_action_verb[0].outputs[0].text

            entry["vlm_output_action_verb"] = vlm_output_action_verb

            messages_objects = [{"role": "system", "content": " "},
                                {"role": "user", "content": [
                                    {"type": "text", "text": "What is the object being assembled in this IKEA assembly video?"},
                                        {"type": "video", 
                                        "video": f"{video_path}",   #/workspace/LLaMA-Factory/data/CCFT_lh_de_513/S04A04I01M0_sshc1dh_16.mp4  /workspace/MiniCPM-o/assets/CCFT/S04A04I01M0_ibacb_2.mp4
                                        "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 28}
                                        ]
                                }]
            prompt_objects = processor.apply_chat_template( messages_objects,
                                                    tokenize=False,
                                                    add_generation_prompt=True,)

            image_inputs, video_inputs, video_kwargs = process_vision_info(messages_objects, return_video_kwargs=True)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            llm_inputs_objects = {"prompt": prompt_objects,
                        "multi_modal_data": mm_data,
                        # FPS will be returned in video_kwargs
                        "mm_processor_kwargs": video_kwargs,}

            outputs_objects = llm.generate([llm_inputs_objects], sampling_params=sampling_params)
            vlm_output_objects = outputs_objects[0].outputs[0].text

            entry["vlm_output_objects"] = vlm_output_objects

            vlm_output = vlm_output_action_verb + vlm_output_objects
            entry["vlm_output"] = vlm_output

            # Parse VLM output
            extracted = parse_vlm_output(vlm_output)
            entry.update({
                "primitive_task": extracted["primitive_task"],
                "action_verb": extracted["action_verb"],
                "object": extracted["object"],
            })

            # update ground truth
            entry.update({
                "primitive_task_gt": primitive_task_gt,
                "action_verb_gt": action_verb_gt,
                "object_gt": object_gt,
            })

            # Calculate similarities
            primitive_task_sim = semantic_similarity(entry["primitive_task"], primitive_task_gt)
            action_verb_sim = semantic_similarity(entry["action_verb"], action_verb_gt)
            object_sim = semantic_similarity(entry["object"], object_gt)

            entry.update({
                "primitive_task_similarity": primitive_task_sim,
                "action_verb_similarity": action_verb_sim,
                "object_similarity": object_sim,
                "is_primitive_task_correct": primitive_task_sim >= similarity_threshold,
                "is_action_verb_correct": action_verb_sim >= similarity_threshold,
                "is_object_correct": object_sim >= similarity_threshold,
            })

            # Update counters
            if entry["is_primitive_task_correct"]:
                correct_primitive_task += 1
            if entry["is_action_verb_correct"]:
                correct_action_verb += 1
            if entry["is_object_correct"]:
                correct_object += 1

            total += 1

        except Exception as e:
            entry["error"] = str(e)
            print(f"Error processing {filename}: {str(e)}")
        
        results_log.append(entry)

    # Calculate metrics
    metrics = {
        "total_samples": total,
        "primitive_task_acc": safe_divide(correct_primitive_task, total),
        "action_verb_acc": safe_divide(correct_action_verb, total),
        "object_acc": safe_divide(correct_object, total),
    }

    # Save log to file
    if log_file_path:
        try:
            full_df = pd.DataFrame(results_log)
            full_df.to_csv(log_file_path, index=False)
            print(f"Full results log saved to {log_file_path}")
            
            # Save wrong predictions
            if wrong_preds_path:
                wrong_preds = full_df[
                    (full_df["is_primitive_task_correct"] == False) |
                    (full_df["is_action_verb_correct"] == False) |
                    (full_df["is_object_correct"] == False) 
                ]
                
                if not wrong_preds.empty:
                    wrong_preds.to_csv(wrong_preds_path, index=False)
                    print(f"Wrong predictions saved to {wrong_preds_path}")
                    print(f"Total wrong predictions: {len(wrong_preds)}/{total}")
                else:
                    print("No wrong predictions to save")
                    
        except Exception as e:
            print(f"Failed to save log files: {str(e)}")

    # Print summary
    print(f"Primitive Task-Wise Accuracy (Threshold={similarity_threshold}): {metrics['primitive_task_acc']:.2%}")
    print("Component-Wise Accuracy:")
    print(f"  Action Verb: {metrics['action_verb_acc']:.2%}")
    print(f"  Object: {metrics['object_acc']:.2%}")

    # Write metrics to a txt file
    with open(txt_file_path, "w") as f:
        f.write(f"Primitive Task-Wise Accuracy (Threshold={similarity_threshold}): {metrics['primitive_task_acc']:.2%}\n")
        f.write("Component-Wise Accuracy:\n")
        f.write(f"  Action Verb: {metrics['action_verb_acc']:.2%}\n")
        f.write(f"  Object: {metrics['object_acc']:.2%}\n")

    return metrics, results_log

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--epoch", type=str, required=True)
    args = parser.parse_args()

    nlp = spacy.load("en_core_web_sm")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    task = args.task
    epoch = args.epoch
    MODEL_PATH = f"/home/hao/CCFT/LLaMA-Factory/models/{task}_{epoch}"
    
    # Load model and processor
    llm = LLM(
        model=MODEL_PATH,
        limit_mm_per_prompt={"image": 10, "video": 10},
    )

    sampling_params = SamplingParams(
        temperature=0.01,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=6144,
            stop_token_ids=[],
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # Run evaluation
    metrics, detailed_log = evaluate_vlm(
        video_folder="/home/hao/CCFT/LLaMA-Factory/data/IKEA_video_clips/test",
        processor=processor,
        llm=llm,
        log_file_path=f"/home/hao/CCFT/Qwen2.5-VL/eval_results/{task}_{epoch}.csv",
        wrong_preds_path=f"/home/hao/CCFT/Qwen2.5-VL/eval_results/{task}_{epoch}_wrong_predictions.csv",
        txt_file_path=f"/home/hao/CCFT/Qwen2.5-VL/eval_results/{task}_{epoch}.txt",
        similarity_threshold=0.95,
    )
    
    del llm, processor