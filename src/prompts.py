"""
Prompt 模板 - 严格按照 COLING 2025 Table 5 & 6
"""


class PromptTemplates:
    """
    Revised Prompts for Priori Judgment (Retrieval-Augmented Setting)
    Optimized for Llama-3 to reproduce baseline performance.
    """
    
    @staticmethod
    def priori_judgment_qa(question: str, context: str) -> str:
        # 修改原因：防止 Llama-3 输出 "Yes, based on..." 废话导致 Unknown 误判
        return (
            f"Refer to the following information and your internal knowledge to answer the question.\n"
            f"If you do not know the answer or the information is insufficient, strictly output \"Unknown\".\n"
            f"Do not output \"Yes\" or \"No\" at the beginning. Just give the answer directly.\n\n"
            f"Information:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer:"
        )
    
    @staticmethod
    def priori_judgment_fact(claim: str, context: str) -> str:
        # 修改原因：强制输出 True/False，适配 src/metrics.py
        return (
            f"Refer to the following information and your internal knowledge to verify the claim.\n"
            f"Output ONLY \"True\" or \"False\".\n"
            f"If you are unsure or the information is insufficient, output \"Unknown\".\n\n"
            f"Information:\n{context}\n\n"
            f"Claim: {claim}\n"
            f"Prediction (True/False/Unknown):"
        )
    
    @staticmethod
    def priori_judgment_truthful(question: str) -> str:
        # 保持简单，防止幻觉
        return (
            f"Q: {question}\n"
            f"A: (If you don't know, say Unknown)"
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
