import comfy.samplers
from .base_node import UtilityNode, NODE_POSTFIX

VALID_SAMPLERS = comfy.samplers.KSampler.SAMPLERS
VALID_SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

class SamplerSelector(UtilityNode):
    """
    Select a sampler from the list of valid samplers.
    If use_override is True, the override_string is used instead of the sampler_name.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (VALID_SAMPLERS,),
            },
            "optional": {
                "override_string": ("STRING", {"multiline": False}),
                "use_override": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (list(VALID_SAMPLERS),)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "select"

    def select(self, sampler_name: str, override_string: str = None, use_override: bool = False):
        if use_override:
            return (override_string,)
        return (sampler_name,)
    
class SchedulerSelector(UtilityNode):
    """
    Select a scheduler from the list of valid schedulers.
    If use_override is True, the override_string is used instead of the scheduler_name.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "scheduler_name": (VALID_SCHEDULERS,),
            },
            "optional": {
                "override_string": ("STRING", {"multiline": False}),
                "use_override": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (list(VALID_SCHEDULERS),)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "select"

    def select(self, scheduler_name: str, override_string: str = None, use_override: bool = False):
        if use_override:
            return (override_string,)
        return (scheduler_name,)
    
class StringToCombo(UtilityNode):
    """
    Convert a string to a combo.
    The input string is **trimmed** and **lower‑cased**.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string": ("STRING", {"multiline": False}),
            },
        }
    
    RETURN_TYPES = ("COMBO","STRING", "*")
    FUNCTION = "string_to_combo"
    
    def string_to_combo(self, string: str):
        return (string.strip().lower(), string, string)
    
# Configuration for node display names

UTILITY_NODE_CLASS_MAPPINGS = {
    "SamplerSelector": SamplerSelector,
    "SchedulerSelector": SchedulerSelector,
    "StringToCombo": StringToCombo,
}

# Generate display names with configurable prefix
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSelector": f"Sampler Selector {NODE_POSTFIX}",
    "SchedulerSelector": f"Scheduler Selector {NODE_POSTFIX}",
    "StringToCombo": f"String to Combo {NODE_POSTFIX}",
}