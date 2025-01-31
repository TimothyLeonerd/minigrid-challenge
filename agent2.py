from openai import OpenAI
from typing import Dict, Any, Tuple, Optional
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from collections import deque
from itertools import count

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

ACTION_MAP = {
    0: "turn left",
    1: "turn right",
    2: "move forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


def relative_to_absolute(agent_direction, relative_direction):
    if agent_direction == "north":
        if relative_direction == "left":
            return "west"
        elif relative_direction == "right":
            return "east"
        elif relative_direction == "front":
            return "north"
    elif agent_direction == "south":
        if relative_direction == "left":
            return "east"
        elif relative_direction == "right":
            return "west"
        elif relative_direction == "front":
            return "south"
    elif agent_direction == "east":
        if relative_direction == "left":
            return "north"
        elif relative_direction == "right":
            return "south"
        elif relative_direction == "front":
            return "east"
    elif agent_direction == "west":
        if relative_direction == "left":
            return "south"
        elif relative_direction == "right":
            return "north"
        elif relative_direction == "front":
            return "west"
    else:
        raise ValueError(f"Invalid agent direction: {agent_direction}")


class Agent:
    def __init__(
        self, api_key: str, model: str = "gpt-4o-mini", api_url: Optional[str] = None
    ):
        """
        Initialize the agent.

        Args:
            api_key: API key
            model: model to use
            temperature: Temperature for model sampling
        """
        self.client = OpenAI(api_key=api_key, base_url=api_url)
        self.model = model
        self.temperature = 0.0
        self.past_states = deque(maxlen=8)  # [state, response]
        self.current_step = 0
        self.current_location = [0, 0]
        self.object_memory = []
        self.object_id_counter = count(1)  # Unique object ID generator
        self.cot_repeat = 6

        # System prompt to explain the task

    def calculate_absolute_position(self, rel_x, rel_y):
        """
        Convert a relative position to an absolute position based on the agent's current location and direction.

        Args:
            rel_x (int): Relative x position
            rel_y (int): Relative y position

        Returns:
            Tuple[int, int]: Absolute (x, y) position
        """
        x, y = self.current_location
        if self.current_direction == "north":
            return x - rel_x, y + rel_y
        elif self.current_direction == "south":
            return x + rel_x, y - rel_y
        elif self.current_direction == "east":
            return x + rel_y, y + rel_x
        elif self.current_direction == "west":
            return x - rel_y, y - rel_x

    def update_memory(self, object_name, object_color, rel_x, rel_y):
        """
        Add a new object to memory if not already tracked.

        Args:
            object_name (str): Name of the object
            object_color (str): Color of the object
            rel_x (int): Relative x position
            rel_y (int): Relative y position
        """
        absolute_position = self.calculate_absolute_position(rel_x, rel_y)

        # Check if object already exists in memory
        for obj in self.object_memory:
            if (
                obj["name"] == object_name
                and obj["color"] == object_color
                and obj["absolute location"] == absolute_position
            ):
                return obj["id"]  # Return existing object ID

        # If not found, assign a new ID
        new_id = next(self.object_id_counter)
        self.object_memory.append(
            {
                "id": new_id,
                "name": object_name,
                "color": object_color,
                "absolute location": absolute_position,
            }
        )
        return new_id

    def update_location(self, action_idx: int, obs: Dict[str, Any]):
        """
        Update the agent's current location based on the action taken.----*-*******

        Args:
            action_idx: The index of the action taken.
        """
        # If the last action was move forward but a wall is in front, don't update location
        if action_idx == 2 and obs["image"][3, 5, 0] in [2, 4, 5, 6, 7]:
            return  # Wall detected, ignore movement
        
        if action_idx == 2:  # Move Forward
            if self.current_direction == "north":
                self.current_location[1] += 1  # Move up (increase y)
            elif self.current_direction == "south":
                self.current_location[1] -= 1  # Move down (decrease y)
            elif self.current_direction == "east":
                self.current_location[0] += 1  # Move right (increase x)
            elif self.current_direction == "west":
                self.current_location[0] -= 1  # Move left (decrease x)

        elif action_idx == 0:  # Turn Left
            if self.current_direction == "north":
                self.current_direction = "west"
            elif self.current_direction == "west":
                self.current_direction = "south"
            elif self.current_direction == "south":
                self.current_direction = "east"
            elif self.current_direction == "east":
                self.current_direction = "north"

        elif action_idx == 1:  # Turn Right
            if self.current_direction == "north":
                self.current_direction = "east"
            elif self.current_direction == "east":
                self.current_direction = "south"
            elif self.current_direction == "south":
                self.current_direction = "west"
            elif self.current_direction == "west":
                self.current_direction = "north"


    def find_last_action(self, action_text, action_map):
        action_idx = None
        last_position = -1
        found_action = None

        # Check each possible action
        for idx, text in action_map.items():
            # Find the last position of this action in the text
            position = action_text.rfind(text)

            # If found and it's later than our previous match
            if position != -1 and position > last_position:
                last_position = position
                action_idx = idx
                found_action = text

        return action_idx, found_action

    def get_CoT_prompt(self):
        return """
        How You Should Think and provide a step-by-step reasoning:
1. Analyze the Environment: Consider your direction, surroundings, walls, and visible objects.
2. Recall Known Objects: Use your memory of previously seen objects and their relation to your current absolute location to decide the best course of action.
3. Follow the Rules: Avoid illegal moves (e.g., moving into a wall, standing still too long, stepping on objects).
4. Plan Your Next Move Determine the best action based on your analysis.
"""

    def get_CoT_prompt_with_math(self):
        return """
        How You Should Think and provide a step-by-step reasoning:
1. Analyze the Environment: Consider your direction, surroundings, walls, and visible objects.
2. Recall Known Objects: Use your memory of previously seen objects and their relation to your current absolute location to decide the best course of action.
3. Follow the Rules: Avoid illegal moves (e.g., moving into a wall, standing still too long, stepping on objects).
4. after reasoning you should have selected an object to head to, then calculate absolut distance from current location and object location, and the rotation difference between current direction and direction from current location to the object, and try to use this information to aid the action decision.
5. Plan Your Next Move Determine the best action based on your analysis.
"""


    def get_system_prompt(self, direction):
        return f"""You are an agent in a grid-world environment. The goal is to navigate the world and interact with objects to complete the mission.

You must choose one of these actions:
- turn left (rotates towards {relative_to_absolute(direction, 'left')})
- turn right (rotates towards {relative_to_absolute(direction, 'right')})
- move forward (moves towards {direction})
- pick up
- drop
- toggle (opens a door with a key or opens a box)

Additional information:
- You can face FOUR different directions: north, south, east, west
- You cannot step on objects, you need to go around them
- Locked doors can be toggled with a key, if they are one cell in front of you
- Keys can be picked up
- Box can contain a key or another object
- Box can be toggled to reveal its content if it's one cell in front of you
- You can pick up and toggle only actionable objects (exactly one cell in front of you)
- If you don't see target object, explore the world to find it.


What should you do next? Conclude with 'Final Action:' followed by your choice."""

# - You can interact (reach, goto) objects by facing them (exactly one cell in front of you).
# - You can face FOUR different directions: north, south, east, west
# - You should focus on the mission object if it was very near to you.
# - You cannot step on objects, you need to go around them
# - If interest object is exactly left or right focus on rotating to correct direction if you need to reach
# - If the path is clear to the wanted interaction object, you should move in optimal distance way.
# - You are not allowed to select forward if you are facing a wall (Wall in front (blocking): yes)
# - You are should not stay in the same place for a long time and repeat old actions.
# - Locked doors can be toggled with a key, if they are one cell in front of you
# - Keys can be picked up
# - Box can contain a key or another object
# - Box can be toggled to reveal its content if it's one cell in front of you
# - You can pick up and toggle only actionable objects (exactly one cell in front of you)
# - If you don't see target object, explore the world to find it.

    def parse_observation(self, obs: Dict[str, Any], mission: str) -> str:
        """
        Convert the observation into a text prompt for the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Formatted prompt string
        """
        # Convert direction number to cardinal direction
        directions = ["east", "south", "west", "north"]
        direction = directions[obs["direction"]]
        self.current_direction = direction
        # Parse the grid to find visible objects
        visible_objects = []
        grid = obs["image"]

        # Convert object types to descriptions
        for x in range(7):
            for y in range(7):
                if x == 3 and y == 6:
                    continue  # skip for agent position - it's the object being held
                obj_id, color_id, door_state = grid[x, y]
                if obj_id > 2:
                    object_name = IDX_TO_OBJECT[obj_id]
                    object_color = IDX_TO_COLOR[color_id]
                    # Calculate relative position
                    rel_x = 3 - x
                    rel_y = 6 - y
                    # Update memory and get object ID
                    obj_memory_id = self.update_memory(object_name, object_color, rel_x, rel_y)
                    obj_state = ""
                    if obj_id == 4:  # it's a door
                        obj_state = f"state: {IDX_TO_STATE[door_state]} "
                    obj_repr = f"\n * [object: {obj_memory_id}] color: {object_color} type: {object_name} {obj_state} is "

                    obj_pos = ""
                    if y < 6:
                        obj_pos += f" {6 - y} cells in the front"
                    if x < 3:
                        if obj_pos != "":
                            obj_pos += " AND"
                        obj_pos += f" {3 - x} cells to the left"
                    elif x > 3:
                        if obj_pos != "":
                            obj_pos += " AND"
                        obj_pos += f" {x - 3} cells to the right"
                    
                    obj_repr = obj_repr + obj_pos
                    visible_objects.append(obj_repr)

        actionable_object = "none"
        if grid[3, 5, 0] > 2:
            actionable_object = (
                f"{IDX_TO_COLOR[grid[3, 5, 1]]} {IDX_TO_OBJECT[grid[3, 5, 0]]}"
            )
        holding_object = "none"
        if grid[3, 6, 0] > 2:
            holding_object = (
                f"{IDX_TO_COLOR[grid[3, 6, 1]]} {IDX_TO_OBJECT[grid[3, 6, 0]]}"
            )

        walls = []
        if grid[2, 6, 0] == 2:
            walls.append(f"left ({relative_to_absolute(direction, 'left')})")
        if grid[4, 6, 0] == 2:
            walls.append(f"right ({relative_to_absolute(direction, 'right')})")
        if grid[3, 5, 0] == 2:
            walls.append(f"front ({relative_to_absolute(direction, 'front')})")
        if len(walls) == 0:
            walls.append("none")

        # Create the prompt
        past_states_str = "\n".join(self.past_states)
        # Format known objects memory
        memory_str = "Known objects:\n" + "\n".join(
            [
                f"[object {obj['id']}] {obj['color']} {obj['name']} at absolute location: {obj['absolute location'][0]} on the x axis and {obj['absolute location'][1]} on the y axis"
                for obj in self.object_memory
            ]
        ) or "none"
        current_state = f"""[Step {self.current_step}]
- Facing '{direction}'
- Absolute location: [{self.current_location[0]} on the x axis and {self.current_location[1]} on the y axis]
- Sensor in the left is detecting: {IDX_TO_OBJECT[grid[2, 6, 1]]} {IDX_TO_OBJECT[grid[2, 6, 0]]} on the left
- Sensor in the right detect: {IDX_TO_OBJECT[grid[4, 6, 1]]} {IDX_TO_OBJECT[grid[4, 6, 0]]} on the right
- Sensor in the front detect: {IDX_TO_OBJECT[grid[3, 5, 1]]} {IDX_TO_OBJECT[grid[3, 5, 0]]} facing you
- Wall in front (blocking): {"yes" if grid[3, 5, 0] == 2 else "no"}
- Visible objects: {', '.join(visible_objects) if visible_objects else 'none'}
- Actionable object: {actionable_object}
- Holding object: {holding_object}
- Mission: {mission}"""
#         current_state = f"""[Step {self.current_step}]
# - Absolute location: [{self.current_location[0]} on the x axis and {self.current_location[1]} on the y axis]
# """

        prompt = f"""Recent states:
{past_states_str}
{memory_str}
{current_state}
Response:"""

        return prompt, current_state, direction

    def get_action(self, obs: Dict[str, Any], mission: str, verbose: bool) -> int:
        """
        Get the next action from the model.

        Args:
            obs: Observation from the environment
            mission: Current mission string

        Returns:
            Action index
        """
        prompt, current_state, direction = self.parse_observation(obs, mission)
        if self.current_step % self.cot_repeat == 0:
            final_prompt = f"{self.get_system_prompt(direction)}\n{self.get_CoT_prompt()}\n{prompt}"
        else:
            final_prompt = f"{self.get_system_prompt(direction)}\n\n{prompt}"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": final_prompt},
            ],
            temperature=self.temperature,
            max_tokens=1000,
        )
        if verbose:
            print("==================================")
            print("final_prompt:\n", final_prompt)
            print("response:\n", response.choices[0].message.content)

        full_response = response.choices[0].message.content.strip().lower()
        # Extract the action after "Final Action:"
        action_idx, action_text = None, None
        if "final action:" in full_response:
            action_part = full_response.split("final action:")[-1].strip()
            action_idx, action_text = self.find_last_action(action_part, ACTION_MAP)
        # action_idx, action_text = self.find_last_action(response, ACTION_MAP)

        if action_idx is None:
            print(
                f"Warning: Invalid action '{action_text}', defaulting to move forward"
            )
            action_idx = 2  # Default to move forward
            action_text = ACTION_MAP[2]

        self.past_states += [
            current_state,
            f"Response: {full_response if self.current_step % self.cot_repeat else action_text}",
        ]
        self.current_step += 1

        # **Update Location and Direction**
        self.update_location(action_idx, obs)
        
        # dict with metadata to log during eval
        metadata = {
            "final_prompt": final_prompt,
            "response": response,
            "action_text": action_text,
        }

        return action_idx, metadata


def handle_state(
    obs: Dict[str, Any], mission: str, agent: Agent, verbose: bool = False
) -> int:
    """
    Process the current state and get the next action.

    Args:
        obs: Current observation from the environment
        mission: Current mission string
        agent: Agent instance
        verbose: Whether to print debug information

    Returns:
        Action index to take
    """

    action, metadata = agent.get_action(obs, mission, verbose)

    if verbose:
        print("Chosen Action:", ACTION_MAP[action])

    return action, metadata
