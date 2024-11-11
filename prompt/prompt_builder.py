from utils.enums import REPR_TYPE
from utils.enums import EXAMPLE_TYPE
from utils.enums import SELECTOR_TYPE
from prompt.PromptReprTemplate import *
from prompt.ExampleFormatTemplate import *
from prompt.ExampleSelectorTemplate import *
from prompt.PromptICLTemplate import BasicICLPrompt


def get_repr_cls(repr_type: str):
    if repr_type == REPR_TYPE.CODE_REPRESENTATION:
        repr_cls = SQLPrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION:
        repr_cls = TextPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION:
        repr_cls = NumberSignPrompt
    elif repr_type == REPR_TYPE.BASIC:
        repr_cls = BaselinePrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT:
        repr_cls = InstructionPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION_WFK:
        repr_cls = NumberSignWithForeignKeyPrompt
    elif repr_type == REPR_TYPE.BASIC_WOFK:
        repr_cls = BaselineWithoutForeignKeyPrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION_WFK:
        repr_cls = TextWithForeignKeyPrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT_WFK:
        repr_cls = InstructionWithForeignKeyPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION_WORULE:
        repr_cls = NumberSignWithoutRulePrompt
    elif repr_type == REPR_TYPE.CODE_REPRESENTATION_WRULE:
        repr_cls = SQLWithRulePrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT_WRULE:
        repr_cls = InstructionWithRulePrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION_WRULE:
        repr_cls = TextWithRulePrompt
    elif repr_type == REPR_TYPE.CODE_REPRESENTATION_COT:
        repr_cls = SQLCOTPrompt
    elif repr_type == REPR_TYPE.TEXT_REPRESENTATION_COT:
        repr_cls = TextCOTPrompt
    elif repr_type == REPR_TYPE.OPENAI_DEMOSTRATION_COT:
        repr_cls = NumberSignCOTPrompt
    elif repr_type == REPR_TYPE.ALPACA_SFT_COT:
        repr_cls = InstructionCOTPrompt
    elif repr_type == REPR_TYPE.CBR:
        repr_cls = CBRPrompt
    else:
        raise ValueError(f"{repr_type} is not supproted yet")
    return repr_cls


def get_example_format_cls(example_format: str):
    if example_format == EXAMPLE_TYPE.ONLY_SQL:
        example_format_cls = SqlExampleStyle
    elif example_format == EXAMPLE_TYPE.QA:
        example_format_cls = QuestionSqlExampleStyle
    elif example_format == EXAMPLE_TYPE.QAWRULE:
        example_format_cls = QuestionSqlWithRuleExampleStyle
    elif example_format == EXAMPLE_TYPE.COMPLETE:
        example_format_cls = CompleteExampleStyle
    elif example_format == EXAMPLE_TYPE.OPENAI_DEMOSTRATION_QA:
        example_format_cls = NumberSignQuestionSqlExampleStyle
    elif example_format == EXAMPLE_TYPE.BASIC_QA:
        example_format_cls = BaselineQuestionSqlExampleStyle
    else:
        raise ValueError(f"{example_format} is not supported yet!")
    return example_format_cls
    

def get_example_selector(selector_type: str):
    if selector_type == SELECTOR_TYPE.COS_SIMILAR:
        selector_cls = CosineSimilarExampleSelector
    elif selector_type == SELECTOR_TYPE.RANDOM:
        selector_cls = RandomExampleSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE:
        selector_cls = EuclideanDistanceExampleSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_THRESHOLD:
        selector_cls = EuclideanDistanceThresholdExampleSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_SKELETON_SIMILARITY_THRESHOLD:
        selector_cls = EuclideanDistanceSkeletonSimilarThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_QUESTION_MASK:
        selector_cls = EuclideanDistanceQuestionMaskSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_PRE_SKELETON_SIMILARITY_THRESHOLD:
        selector_cls = EuclideanDistancePreSkeletonSimilarThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_PRE_SKELETON_SIMILARITY_PLUS:
        selector_cls = EuclideanDistancePreSkeletonSimilarPlusSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD:
        selector_cls = EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdSelector
    elif selector_type == SELECTOR_TYPE.EUC_DISTANCE_MASK_PRE_SKELETON_SIMILARITY_THRESHOLD_SHIFT:
        selector_cls = EuclideanDistanceQuestionMaskPreSkeletonSimilarThresholdShiftSelector
    else:
        raise ValueError(f"{selector_type} is not supported yet!")
    return selector_cls
    

def prompt_factory(repr_type:str, k_shot: int, example_format: str, selector_type: str):
    # 根據 repr_type 獲取表示類別（representation class）
    repr_cls = get_repr_cls(repr_type)

    if k_shot == 0:  # 當 k_shot = 0 時，創建零範例學習的 Prompt 類別
        assert repr_cls is not None # 確保表示類別已定義
        cls_name = f"{repr_type}_{k_shot}-SHOT" # 設定類別名稱，表示零範例學習
        # 定義 Prompt 類別，繼承 repr_cls 和 BasicICLPrompt
        class PromptClass(repr_cls, BasicICLPrompt):
            name = cls_name  # 設置類別名稱
            NUM_EXAMPLE = k_shot # 設置範例數量為 0
            def __init__(self, *args, **kwargs): # 類別初始化函數
                repr_cls.__init__(self, *args, **kwargs) # 初始化表示類別
                # init tokenizer
                BasicICLPrompt.__init__(self, *args, **kwargs) # 初始化 BasicICLPrompt 類別
    else: # 當 k_shot > 0 時，創建多範例學習的 Prompt 類別
        # 根據 example_format 和 selector_type 獲取相應的類別
        example_format_cls = get_example_format_cls(example_format) # 獲取範例格式類別
        selector_cls = get_example_selector(selector_type) # 獲取範例選擇器類別
        cls_name = f"{repr_type}_{k_shot}-SHOT_{selector_type}_{example_format}-EXAMPLE" # 設定類別名稱，包含 k_shot、選擇器、範例格式等信息
        # 定義 Prompt 類別，繼承 selector_cls、example_format_cls、repr_cls 和 BasicICLPrompt
        class PromptClass(selector_cls, example_format_cls, repr_cls, BasicICLPrompt):
            name = cls_name # 設置類別名稱
            NUM_EXAMPLE = k_shot # 設置範例數量
            # 類別初始化函數
            def __init__(self, *args, **kwargs):
                selector_cls.__init__(self, *args, **kwargs) # 初始化選擇器類別
                # init tokenizer
                BasicICLPrompt.__init__(self, *args, **kwargs) # 初始化 BasicICLPrompt 類別
    
    return PromptClass # 返回動態創建的 Prompt 類別