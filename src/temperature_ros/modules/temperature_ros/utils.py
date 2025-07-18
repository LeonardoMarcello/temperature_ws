

from typing import List, Callable
from collections import Counter
from vision_msgs.msg import Detection2D, Detection2DArray
#import pyfiglet
from tqdm import tqdm

def sleep_with_progressbar(seconds: int, sleep_handle: Callable, description: str):
    # Sleep for n seconds with a progress bar
    for _ in tqdm(range(seconds), desc=description, ncols=100, unit="s", bar_format="{l_bar}{bar} {remaining} seconds left"):
        sleep_handle(1)  # Sleep for 1 second per iteration


"""
def print_ascii_art(text:str):
    ascii_art = pyfiglet.figlet_format(text)
    
    # Get the width of the longest line in the ASCII art
    lines = ascii_art.splitlines()
    max_width = max(len(line) for line in lines)

    # Create the top and bottom borders of the window
    border = "+" + "-" * (max_width + 2) + "+"

    # Add the window around the ASCII art
    windowed_text = border + "\n"
    for line in lines:
        # Right-pad each line with spaces to ensure it matches the max width
        padded_line = line.ljust(max_width)
        windowed_text += "| " + padded_line + " |\n"
    windowed_text += border

    print(windowed_text)
"""

def parse_classes_file(path):
    classes = []
    with open(path, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            classes.append(line)
    return classes



def pop_prediction(detection: Detection2D, classes: List, bound: List, who: List) -> bool:
    # Crope element otuside interesting area or bad id
    center = detection.bbox.center
    object = classes[detection.results[0].id]

    if (bound[1] < center.x  or center.x < bound[0]) or \
       (bound[3] < center.y or center.y < bound[2]) or \
       (object not in who): 
        return True
    else: 
        return False 
    
def print_detections(detections: Detection2DArray, classes: List):
    detections_counter = Counter(detection.results[0].id for detection in detections)

    # Print counts
    for id, count in detections_counter.items():
        print(f"Detected {count} object of class {classes[id]}")


def check_detections(detections: Detection2DArray, classes: List, who: List) -> bool :
    detections_counter = Counter(detection.results[0].id for detection in detections)

    # Print counts
    for id, count in detections_counter.items():
        if (classes[id] in who) and count>0: 
            return True
    return False