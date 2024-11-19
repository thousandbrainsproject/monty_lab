# touch-sensor project
NOTE: This is integrated as the surface agent in the Monty code base.

## Monty Experiment description

### Agents
- There is one agent (The Finger) with two sensors
- The 'view_finder' sensor is at the finger base and shows an rgb image
- The 'patch' sensor is at the finger tip, where the agent is also located. It shows a depth image and measures object features

### Actions
- **move_forward**: default action, moves agent forward along the axis it is facing.
- **orient_vertical** (custom action): rotate the agent in the up-down axis, to face the object from a different direction (typically, perpendicular to the point normal), *without changing the point on the object it is fixated on*. This latter aspect is achieved through two translation actions that compensate for the turning action.
- **orient_horizontal** (custom action): does that same as orient_vertical, but in the left-right axis (in the agent's reference frame) rather than in the up-down axis.
- **move_tangentially** (custom action): move tangentially to the plane that the agent is facing (typically, this is called when the agent plane is parallel to the object plane it is facing). The direction it moves along is specified by an angle between 0 - 2pi.

### Experiment
- A stripped down experiment that takes a specified number of steps in the environment, and then plots the view_finder and patch observations in a matplotlib window.

### Policy
- Includes a **touch_object** method. This initializes the episode by getting the depth from the object and moving the agent to the *desired_object_distance*.
- Cycles through four actions, one after the other: move forward -> orient right -> orient up -> move tangentially
- **orienting_angle_from_normal** takes the point normal feature observation from the touch sensor and returns the number of degrees to turn in order to be parallel to that vector.
- The three custom actions require not only a scalar amount (*ActuationSpec.amount*) but also additional values. *move_tangentially* requires the angular direction. The orienting actions require a forward translation and a leftward (in the case or orient_horizontal) or downward (in the case of orient_vertical) translation such that the agent maintains fixation on the same part of the object even after turning. These additional values are stored in the *ActuationSpec.constraint* variable, which gets passed to the action function. These constraints are first computed using the method **get_next_constraint**.
- Note: the touch sensor must sense at least one pixel on the object, otherwise an error will be thrown and the experiment will end

### Sensor module
- Added a depth feature to the sensor module. This is used to calculate the constraints for the orienting actions
- Adds an object_coverage feature with the proportion of sensor that detects the object

### Transform 
- Added a clip function to the transform. This clips the patch sensor's depth camera at the clip_value and sets all depth readings > clip_value as equal to clip_value. It also sets all semantic values at depths > clip_value to zero (no object detected)

### Notable config args that I have specified
- In **config_args.py**, I specify: the action space above, *desired_object_distance* (0.025), action amount for *move_tangentially* (0.004), and the sensor module configuration.
- In **dataset_configs.py**, I specify: the *clip_value* (0.04), the agent's starting position ([0.0, 1.5, 0.1]), the sensor resolutions ([[8, 8], [256, 256]]), the sensor positions relative to the agent ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.03]]), and the sensor camera zoom values ([6.0, 1.0]).
- In **touch_sensor_experiments.py**, I specify: *max_train_steps* (40), YCB object indices ([2,3,5,9,11])

## Running an experiment
*Note: This experiment requires Habitat_sim v0.2.2_rc1, which allows us to change the height of the agent to zero and remove the floor from the habitat environment. Luiz is working on getting us a version of this; otherwise, you can install it from the habitat_sim github directly. Contact me or Luiz for details.*
```
python projects/touch_sensor/run.py
```