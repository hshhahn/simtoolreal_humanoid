"""Helper modules for the SimToolReal task.

Organized into:
  - ``generate_objects`` — procedural URDF emitter for handle+head primitives.
  - ``object_size_distributions`` — per-type size ranges driving the generator.
  - ``scene_utils`` — robot/table cfg builders, joint PD tables, material binding.
  - ``obs_utils`` — obs-field sizes + stack helper.
  - ``reward_utils`` — pure-torch reward term helpers.
  - ``logging_utils`` — task extras published for training logs.
"""
