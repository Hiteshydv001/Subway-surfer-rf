import cv2
import numpy as np
import os
import config # Import config

def load_templates(template_dir=config.TEMPLATE_DIR):
    """Loads all .png templates from the specified directory."""
    templates = {}
    if not os.path.isdir(template_dir):
        raise FileNotFoundError(f"Template directory not found: {template_dir}")

    print(f"Loading templates from: {template_dir}")
    loaded_files = []
    for filename in os.listdir(template_dir):
        if filename.lower().endswith(".png"):
            name = os.path.splitext(filename)[0] # Use filename without extension as key
            path = os.path.join(template_dir, filename)
            # Load as grayscale directly
            template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template_img is None:
                print(f"Warning: Could not load template image: {path}")
            else:
                # Ensure template name exists in config OBSTACLE_TYPES or is 'game_over'/'start_game'
                if name in config.OBSTACLE_TYPES or name in ['game_over', 'start_game']:
                    templates[name] = template_img
                    loaded_files.append(name)
                else:
                     print(f"Warning: Template file '{filename}' does not have a corresponding entry "
                           f"in config.OBSTACLE_TYPES and is not 'game_over' or 'start_game'. Ignoring.")

    print(f"  Loaded templates: {loaded_files}")
    if not templates:
        raise ValueError(f"No valid templates loaded from {template_dir}. Check config.py and filenames.")
    return templates

def match_template(image_gray, template, threshold=config.TEMPLATE_MATCH_THRESHOLD, method=cv2.TM_CCOEFF_NORMED):
    """
    Finds all occurrences of a template in an image above a threshold
    using Non-Maximum Suppression (NMS) to reduce overlapping boxes.

    Args:
        image_gray (numpy.ndarray): Grayscale image to search within.
        template (numpy.ndarray): Grayscale template image to find.
        threshold (float): Minimum matching confidence (0.0 to 1.0).
        method (int): OpenCV template matching method.

    Returns:
        list: Tuples of ((x, y), w, h, confidence) for each non-overlapping match.
    """
    if template is None or image_gray is None: return []
    if template.shape[0] > image_gray.shape[0] or template.shape[1] > image_gray.shape[1]: return []

    h, w = template.shape # Template dimensions (height, width)

    try:
        res = cv2.matchTemplate(image_gray, template, method)
    except cv2.error as e:
        print(f"OpenCV error during matchTemplate: {e} (Image: {image_gray.shape}, Template: {template.shape})")
        return []

    # Get locations where the match exceeds the threshold
    loc = np.where(res >= threshold)
    # Store potential matches as rectangles with confidence: [x, y, x+w, y+h, confidence]
    rectangles = []
    for pt in zip(*loc[::-1]): # pt is (x, y) - top-left corner
        confidence = res[pt[1], pt[0]]
        rectangles.append([pt[0], pt[1], pt[0] + w, pt[1] + h, confidence])

    # Apply Non-Maximum Suppression (basic version)
    # A more robust NMS could use OpenCV's dnn.NMSBoxes or a dedicated library function
    # if performance becomes an issue with many overlapping detections.
    if not rectangles: return []

    rectangles = np.array(rectangles)
    # Sort by confidence (highest first)
    indices = np.argsort(rectangles[:, 4])[::-1]

    final_matches = []
    processed_indices = set()

    for i in indices:
        if i in processed_indices:
            continue

        # Keep this rectangle
        current_rect = rectangles[i]
        final_matches.append( ( (int(current_rect[0]), int(current_rect[1])), # Top-left (x, y)
                                w, h, current_rect[4] ) )                      # width, height, confidence
        processed_indices.add(i)

        # Find overlapping rectangles
        for j in indices:
            if j in processed_indices:
                continue

            other_rect = rectangles[j]
            # Calculate Intersection over Union (IoU) or simpler overlap check
            # Simple overlap area check:
            x_overlap = max(0, min(current_rect[2], other_rect[2]) - max(current_rect[0], other_rect[0]))
            y_overlap = max(0, min(current_rect[3], other_rect[3]) - max(current_rect[1], other_rect[1]))
            overlap_area = x_overlap * y_overlap
            if overlap_area > 0: # Or use IoU > threshold (e.g., 0.3)
                processed_indices.add(j) # Suppress this overlapping rectangle

    return final_matches