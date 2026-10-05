from typing import Dict, List

# ── Full DexToolBench data structure ──────────────────────────────────────────
# {object_category: {object_name: [task_name, ...]}}
DEXTOOLBENCH_DATA_STRUCTURE: Dict[str, Dict[str, List[str]]] = {
    "hammer": {
        "claw_hammer": ["swing_down", "swing_side"],
        "mallet_hammer": ["swing_down", "swing_side"],
    },
    "marker": {
        "sharpie_marker": ["draw_smile", "write_c"],
        "staples_marker": ["draw_smile", "write_c"],
    },
    "eraser": {
        "flat_eraser": ["wipe_smile", "wipe_c"],
        "handle_eraser": ["wipe_smile", "wipe_c"],
    },
    "brush": {
        "blue_brush": ["sweep_forward", "sweep_right"],
        "red_brush": ["sweep_forward", "sweep_right"],
    },
    "spatula": {
        "flat_spatula": ["serve_plate", "flip_over"],
        "spoon_spatula": ["serve_plate", "flip_over"],
    },
    "screwdriver": {
        "long_screwdriver": ["spin_vertical", "spin_horizontal"],
        "short_screwdriver": ["spin_vertical", "spin_horizontal"],
    },
}

ALL_OBJECT_CATEGORIES = sorted(DEXTOOLBENCH_DATA_STRUCTURE.keys())
ALL_OBJECT_NAMES = sorted(
    object_name
    for object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.values()
    for object_name in object_name_to_task_names.keys()
)
ALL_TASK_NAMES = sorted(
    set(
        task_name
        for object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.values()
        for task_names in object_name_to_task_names.values()
        for task_name in task_names
    )
)

OBJECT_NAME_TO_CATEGORY: Dict[str, str] = {
    object_name: object_category
    for object_category, object_name_to_task_names in DEXTOOLBENCH_DATA_STRUCTURE.items()
    for object_name in object_name_to_task_names
}
