"""
Prompt 模板 - 严格按照 COLING 2025 Table 5 & 6
"""


class PromptTemplates:
    """论文标准 Prompt"""
    
    @staticmethod
    def priori_judgment_qa(question: str, context: str) -> str:
        """
        Priori Judgment for Open-domain QA / Long-form QA
        
        来源：COLING 2025 Table 6 - 第一行
        """
        return (
            f"Given the following information:\n"
            f"{context}\n\n"
            f"Can you answer the following question based on the given information "
            f"or your internal knowledge? If yes, you should give a short answer with "
            f"one or few words, if no, you should answer \"Unknown\".\n\n"
            f"Question: {question}"
        )
    
    @staticmethod
    def priori_judgment_fact(claim: str, context: str) -> str:
        """
        Priori Judgment for Fact Checking
        
        来源：COLING 2025 Table 6 - 第二行
        """
        return (
            f"Given the following information:\n"
            f"{context}\n\n"
            f"Can you verify the following claim based on the given information or "
            f"your internal knowledge? If yes, give a short answer with one or few words. "
            f"If not, answer \"Unknown\".\n\n"
            f"Claim: {claim}"
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