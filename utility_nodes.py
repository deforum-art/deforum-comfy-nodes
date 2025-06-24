import comfy.samplers
import torch
import random
from .tools import VariantSupport
from .base_node import NODE_POSTFIX, ListNode, LogicNode, FlowNode, DebugNode, UtilityNode

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
    
    
@VariantSupport()
class AccumulateNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "to_add": ("*",),
            },
            "optional": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "accumulate"
    
    def accumulate(self, to_add, accumulation = None):
        if accumulation is None:
            value = [to_add]
        else:
            value = accumulation["accum"] + [to_add]
        return ({"accum": value},)

@VariantSupport()
class AccumulationHeadNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_head"

    def accumulation_head(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (accumulation, None)
        else:
            return ({"accum": accum[1:]}, accum[0])

@VariantSupport()
class AccumulationTailNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION", "*",)
    FUNCTION = "accumulation_tail"

    def accumulation_tail(self, accumulation):
        accum = accumulation["accum"]
        if len(accum) == 0:
            return (None, accumulation)
        else:
            return ({"accum": accum[:-1]}, accum[-1])

@VariantSupport()
class AccumulationToListNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("*",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "accumulation_to_list"

    def accumulation_to_list(self, accumulation):
        return (accumulation["accum"],)

@VariantSupport()
class ListToAccumulationNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    INPUT_IS_LIST = True
    FUNCTION = "list_to_accumulation"

    def list_to_accumulation(self, list):
        return ({"accum": list},)

@VariantSupport()
class AccumulationGetLengthNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "accumlength"

    def accumlength(self, accumulation):
        return (len(accumulation['accum']),)
        
@VariantSupport()
class AccumulationGetItemNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1})
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "get_item"

    def get_item(self, accumulation, index):
        return (accumulation['accum'][index],)
        
@VariantSupport()
class AccumulationSetItemNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "accumulation": ("ACCUMULATION",),
                "index": ("INT", {"default":0, "step":1}),
                "value": ("*",),
            },
        }

    RETURN_TYPES = ("ACCUMULATION",)
    FUNCTION = "set_item"

    def set_item(self, accumulation, index, value):
        new_accum = accumulation['accum'][:]
        new_accum[index] = value
        return ({"accum": new_accum},)

@VariantSupport()
class DebugPrint(DebugNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("*",),
                "label": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "debug_print"

    def debugtype(self, value):
        if isinstance(value, list):
            result = "["
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += "]"
        elif isinstance(value, tuple):
            result = "("
            for i, v in enumerate(value):
                result += (self.debugtype(v) + ",")
            result += ")"
        elif isinstance(value, dict):
            result = "{"
            for k, v in value.items():
                result += ("%s: %s," % (self.debugtype(k), self.debugtype(v)))
            result += "}"
        elif isinstance(value, str):
            result = "'%s'" % value
        elif isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            result = str(value)
        elif isinstance(value, torch.Tensor):
            result = "Tensor[%s]" % str(value.shape)
        else:
            result = type(value).__name__
        return result

    def debug_print(self, value, label):
        print("[%s]: %s" % (label, self.debugtype(value)))
        return (value,)

NUM_LIST_SOCKETS = 10
@VariantSupport()
class MakeListNode(ListNode):
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value1": ("*",),
            },
            "optional": {
                "value%d" % i: ("*",) for i in range(1, NUM_LIST_SOCKETS)
            },
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "make_list"
    OUTPUT_IS_LIST = (True,)

    def make_list(self, **kwargs):
        result = []
        for i in range(NUM_LIST_SOCKETS):
            if "value%d" % i in kwargs:
                result.append(kwargs["value%d" % i])
        return (result,)

@VariantSupport()
class GetFloatFromList(ListNode):
    """
    Get the float from the list at the index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("FLOAT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "get_float_from_list"
    
    def get_float_from_list(self, list: list, index: int):
        return (list[index],)
    
@VariantSupport()
class GetIntFromList(ListNode):
    """
    Get the int from the list at the index.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "list": ("INT",),
                "index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            },
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_int_from_list"
    
    def get_int_from_list(self, list: list, index: int):
        return (list[index],)


@VariantSupport()
class IntegerListGeneratorNode(ListNode):
    """
    Generate a list of integer values based on starting integer and control mode.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_int": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "step": 1}),
                "quantity": ("INT", {"default": 5, "min": 1, "max": 10000, "step": 1}),
                "control_mode": (["increment", "random"],),
            },
        }
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_integer_list"
    
    def generate_integer_list(self, start_int: int, quantity: int, control_mode: str):
        integers = []
        
        if control_mode == "increment":
            for i in range(quantity):
                integers.append(start_int + i)
        elif control_mode == "random":
            # Use start_int as master seed for deterministic random generation
            random.seed(start_int)
            for i in range(quantity):
                if i == 0:
                    # First seed is always the start seed
                    integers.append(start_int)
                else:
                    # Generate random seeds using the master seed
                    integers.append(random.randint(0, 0xffffffff))
        
        return (integers,)
    
# Configuration for node display names

UTILITY_NODE_CLASS_MAPPINGS = {
    f"SamplerSelector{NODE_POSTFIX}": SamplerSelector,
    f"SchedulerSelector{NODE_POSTFIX}": SchedulerSelector,
    f"StringToCombo{NODE_POSTFIX}": StringToCombo,
    f"AccumulateNode{NODE_POSTFIX}": AccumulateNode,
    f"AccumulationHeadNode{NODE_POSTFIX}": AccumulationHeadNode,
    f"AccumulationTailNode{NODE_POSTFIX}": AccumulationTailNode,
    f"AccumulationToListNode{NODE_POSTFIX}": AccumulationToListNode,
    f"ListToAccumulationNode{NODE_POSTFIX}": ListToAccumulationNode,
    f"AccumulationGetLengthNode{NODE_POSTFIX}": AccumulationGetLengthNode,
    f"AccumulationGetItemNode{NODE_POSTFIX}": AccumulationGetItemNode,
    f"AccumulationSetItemNode{NODE_POSTFIX}": AccumulationSetItemNode,
    f"DebugPrint{NODE_POSTFIX}": DebugPrint,
    f"MakeListNode{NODE_POSTFIX}": MakeListNode,
    f"GetFloatFromList{NODE_POSTFIX}": GetFloatFromList,
    f"GetIntFromList{NODE_POSTFIX}": GetIntFromList,
    f"IntegerListGeneratorNode{NODE_POSTFIX}": IntegerListGeneratorNode,
}

# Generate display names with configurable prefix
UTILITY_NODE_DISPLAY_NAME_MAPPINGS = {
    f"SamplerSelector{NODE_POSTFIX}": f"Sampler Selector {NODE_POSTFIX}",
    f"SchedulerSelector{NODE_POSTFIX}": f"Scheduler Selector {NODE_POSTFIX}",
    f"StringToCombo{NODE_POSTFIX}": f"String to Combo {NODE_POSTFIX}",
    f"AccumulateNode{NODE_POSTFIX}": f"Accumulate {NODE_POSTFIX}",
    f"AccumulationHeadNode{NODE_POSTFIX}": f"Accumulation Head {NODE_POSTFIX}",
    f"AccumulationTailNode{NODE_POSTFIX}": f"Accumulation Tail {NODE_POSTFIX}",
    f"AccumulationToListNode{NODE_POSTFIX}": f"Accumulation to List {NODE_POSTFIX}",
    f"ListToAccumulationNode{NODE_POSTFIX}": f"List to Accumulation {NODE_POSTFIX}",
    f"AccumulationGetLengthNode{NODE_POSTFIX}": f"Accumulation Get Length {NODE_POSTFIX}",
    f"AccumulationGetItemNode{NODE_POSTFIX}": f"Accumulation Get Item {NODE_POSTFIX}",
    f"AccumulationSetItemNode{NODE_POSTFIX}": f"Accumulation Set Item {NODE_POSTFIX}",
    f"DebugPrint{NODE_POSTFIX}": f"Debug Print {NODE_POSTFIX}",
    f"MakeListNode{NODE_POSTFIX}": f"Make List {NODE_POSTFIX}",
    f"GetFloatFromList{NODE_POSTFIX}": f"Get Float From List {NODE_POSTFIX}",
    f"GetIntFromList{NODE_POSTFIX}": f"Get Int From List {NODE_POSTFIX}",
    f"IntegerListGeneratorNode{NODE_POSTFIX}": f"Integer List Generator {NODE_POSTFIX}",
}