import numpy as np
import config
from .template_matcher import match_template # Use the function from the same directory

def classify_lane(x_center, screen_width):
    """Classifies an x-coordinate into one of three lanes (0, 1, 2)."""
    lane_width = screen_width / 3.0
    if x_center < lane_width: return 0   # Left
    elif x_center < 2 * lane_width: return 1 # Middle
    else: return 2   # Right

def extract_state(screen_gray, object_templates):
    """
    Extracts state: the type of the *closest* detected object in the
    danger zone for each lane.

    Args:
        screen_gray (numpy.ndarray): Grayscale game screen.
        object_templates (dict): Templates for game objects (trains, barriers, coins).

    Returns:
        numpy.ndarray: State vector [lane0_type, lane1_type, lane2_type]
                       using type IDs from config.OBSTACLE_TYPES.
                       Returns None if screen is invalid.
    """
    if screen_gray is None: return None

    screen_height, screen_width = screen_gray.shape
    danger_zone_y_pixel_start = int(screen_height * config.DANGER_ZONE_Y_START)
    danger_zone_y_pixel_end = int(screen_height * config.DANGER_ZONE_Y_END)

    # State: [closest_type_lane0, closest_type_lane1, closest_type_lane2]
    # Initialize with 'clear' type and max y-distance (bottom of screen)
    lane_closest_obstacle = [
        {"type": config.OBSTACLE_TYPES["clear"], "y_bottom": screen_height + 1},
        {"type": config.OBSTACLE_TYPES["clear"], "y_bottom": screen_height + 1},
        {"type": config.OBSTACLE_TYPES["clear"], "y_bottom": screen_height + 1}
    ]

    # Iterate through object templates provided
    for template_name, template_img in object_templates.items():
        obstacle_type_id = config.OBSTACLE_TYPES.get(template_name)
        # Should not happen if templates are loaded correctly, but check anyway
        if obstacle_type_id is None or obstacle_type_id == config.OBSTACLE_TYPES["clear"]:
            continue

        # Find all non-overlapping matches for this template
        matches = match_template(screen_gray, template_img, threshold=config.TEMPLATE_MATCH_THRESHOLD)

        for (x, y), w, h, confidence in matches:
            match_bottom_y = y + h # Use bottom edge for proximity check

            # Check if the *bottom* of the obstacle is within the vertical danger zone
            if danger_zone_y_pixel_start <= match_bottom_y <= danger_zone_y_pixel_end:
                x_center = x + w / 2
                lane_index = classify_lane(x_center, screen_width)

                # If this obstacle is closer (higher on screen = smaller y) than
                # the current closest one in this lane, update the state for that lane.
                if match_bottom_y < lane_closest_obstacle[lane_index]["y_bottom"]:
                    lane_closest_obstacle[lane_index]["type"] = obstacle_type_id
                    lane_closest_obstacle[lane_index]["y_bottom"] = match_bottom_y
                    # Debug: print(f"  Update Lane {lane_index}: {template_name} at y={match_bottom_y}")


    # Final state is the array of types of the closest obstacles found
    state_vector = np.array([lane["type"] for lane in lane_closest_obstacle], dtype=np.int32)
    # print(f"Extracted State: {state_vector}") # Debug
    return state_vector