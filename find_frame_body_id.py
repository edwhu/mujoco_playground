#!/usr/bin/env python3
"""Script to find the frame body ID in the door_scene.xml setup."""

import mujoco
import numpy as np
from pathlib import Path

def find_frame_body_id():
    """Find the frame body ID in the door_scene.xml model."""
    
    # Path to the door_scene.xml file
    xml_path = Path("mujoco_playground/_src/manipulation/leap_hand/xmls/door_scene.xml")
    
    if not xml_path.exists():
        print(f"Error: XML file not found at {xml_path}")
        print("Please run this script from the workspace root directory.")
        return None
    
    try:
        # Load the model
        print(f"Loading model from: {xml_path}")
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        
        # Search for the frame body by name
        frame_body_id = None
        frame_body_name = None
        
        print("\nSearching for frame body...")
        print("Available bodies:")
        
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                print(f"  Body {i}: {body_name}")
                if body_name == "frame":
                    frame_body_id = i
                    frame_body_name = body_name
            else:
                print(f"  Body {i}: <unnamed>")
        
        if frame_body_id is not None:
            print(f"\n✅ Found frame body!")
            print(f"   Name: {frame_body_name}")
            print(f"   ID: {frame_body_id}")
            print(f"\nYou can update the _get_frame_body_id() function to return {frame_body_id}")
        else:
            print(f"\n❌ Frame body not found!")
            print("The frame body might have a different name or might not be loaded properly.")
            
            # Let's also check if there are any bodies with "frame" in the name
            print("\nBodies containing 'frame' in name:")
            for i in range(model.nbody):
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name and "frame" in body_name.lower():
                    print(f"  Body {i}: {body_name}")
        
        # Also show some additional useful information
        print(f"\nModel statistics:")
        print(f"  Total bodies: {model.nbody}")
        print(f"  Total joints: {model.njnt}")
        print(f"  Total actuators: {model.nu}")
        print(f"  Total sites: {model.nsite}")
        
        return frame_body_id
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    frame_id = find_frame_body_id()
    if frame_id is not None:
        print(f"\n🎯 Frame body ID: {frame_id}")
    else:
        print("\n❌ Could not determine frame body ID") 