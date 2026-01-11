"""
Prompt 模板 - 严格按照 COLING 2025 Table 5 & 6
"""

class PromptTemplates:
    """
    Strict Reproduction of Prompts from COLING 2025 Table 6
    """
    
    @staticmethod
    def priori_judgment_qa(question: str, context: str) -> str:
        """
        适用于 NQ, TriviaQA, WebQA
        来源: COLING 2025 原文代码 (utils/prompt.py - 'ra' key)
        特点: 逗号连接，全小写 if yes/no
        """
        return (
            f"Given the following information: \n" # 注意原文这里是 \n 不是 \n\n
            f"{context}\n"
            f"Can you answer the following question based on the given information or your internal knowledge, " # 逗号
            f"if yes, you should give a short answer with one or few words, " # 逗号
            f"if no, you should answer \"Unknown\".\n" # 句号
            f"Question: {question}"
        )
    
    @staticmethod
    def priori_judgment_fact(claim: str, context: str) -> str:
        """
        适用于 FactKG
        来源: COLING 2025 Table 6 (Fact Checking - Retrieval Augmented)
        """
        return (
            f"Given the following information:\n"
            f"{context}\n\n"
            f"Can you verify the following claim based on the given information or your internal knowledge? "
            f"If yes, you should answer True or False, if no, you should answer \"Unknown\".\n\n"
            f"Claim: {claim}"
        )
    
    @staticmethod
    def priori_judgment_truthful(question: str, context: str) -> str:
        """
        适用于 TruthfulQA
        注意：CARE 论文中 TruthfulQA 也使用了 RAG 上下文，Prompt 结构与 QA 类似。
        """
        return (
            f"Given the following information:\n"
            f"{context}\n\n"
            f"Can you answer the following question based on the given information or your internal knowledge? "
            f"If yes, you should give a short answer with one or few words, if no, you should answer \"Unknown\".\n\n"
            f"Question: {question}"
        )
    
    @staticmethod
    def closedbook_qa_short(question: str) -> str:
        """
        Closed-book for short-form QA
        
        来源：COLING 2025 Table 5 - Open-Domain QA Closed-Book
        """
        return (
            f"Answer the questions:\n"
            f"Question: {question}?\n"
            f"The answer is:"
        )
    
    @staticmethod
    def closedbook_qa_long(question: str) -> str:
        """
        Closed-book for long-form QA
        
        来源：COLING 2025 Table 5 - Long-form QA Closed-Book
        """
        return (
            f"Answer the questions:\n"
            f"Question: {question}\n"
            f"The answer is:"
        )
    
    @staticmethod
    def closedbook_fact(claim: str) -> str:
        """
        Closed-book for fact checking
        
        来源：COLING 2025 Table 5 - Fact Checking Closed-Book
        """
        return (
            f"Verify the following claims with \"True\" or \"False\":\n"
            f"Claim: {claim}\n"
            f"The answer is:"
        )
