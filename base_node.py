NODE_NAME = "Deforum"  # Can be changed to any desired prefix

NODE_POSTFIX = "| Deforum"

# Base class for all nodes
class BaseNode:
    CATEGORY = NODE_NAME
    
    @classmethod
    def get_category(cls):
        return cls.CATEGORY
    
class LogicNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Logic"
    
class FlowNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Flow"
    
class UtilityNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Utility"
    
class DebugNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Debug"
    
class ListNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Lists"
    
class LatentNode(BaseNode):
    CATEGORY = f"{NODE_NAME}/Latent"