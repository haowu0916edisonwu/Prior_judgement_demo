"""
Prompt 模板 - 混合 SOTA 版 (Hybrid V5 - Precision)
---------------------------------------------------------
1. QA (NQ/WebQA/TriviaQA): CARE 格式 ("If yes, give a short answer...")
2. TruthfulQA: COLING 格式 ("you should give a detailed answer")
3. FactKG: COLING 原版。
"""

class PromptTemplates:
    
    @staticmethod
    def priori_judgment_qa(question: str, context: str) -> str:
        """[CARE EMNLP 2025] 适用于 NQ, TriviaQA, WebQA"""
        return (
            f"Given the following information:\n"
            f"{context}\n"
            f"Can you answer the following question based on the given information or your internal knowledge? "
            f"If yes, give a short answer with one or few words. "
            f"If not, answer \"Unknown\".\n"
            f"Question: {question}\n"
            f"Answer:" 
        )

    @staticmethod
    def priori_judgment_truthful(question: str, context: str) -> str:
        """[COLING 2025 回滚版] 适用于 TruthfulQA"""
        return (
            f"Given the following information: \n" 
            f"{context}\n"
            f"Can you answer the following question based on the given information or your internal knowledge, " 
            f"if yes, you should give a detailed answer, " 
            f"if no, you should answer \"Unknown\".\n" 
            f"Question: {question}\n"
            f"Answer:" 
        )

    @staticmethod
    def priori_judgment_fact(claim: str, context: str) -> str:
        """[COLING 2025 原版] 适用于 FactKG"""
        return (
            f"Given the following information:\n"
            f"{context}\n\n"
            f"Can you verify the following claim based on the given information or your internal knowledge? "
            f"If yes, you should answer True or False, if no, you should answer \"Unknown\".\n\n"
            f"Claim: {claim}\n"
            f"Answer:"
        )
        
    @staticmethod
    def closedbook_qa_short(question: str) -> str:
        return (
            f"Answer the questions:\n"
            f"Question: {question}?\n"
            f"The answer is:"
        )
    
    @staticmethod
    def closedbook_qa_long(question: str) -> str:
        return (
            f"Answer the questions:\n"
            f"Question: {question}\n"
            f"The answer is:"
        )
    
    @staticmethod
    def closedbook_fact(claim: str) -> str:
        return (
            f"Verify the following claims with \"True\" or \"False\":\n"
            f"Claim: {claim}\n"
            f"The answer is:"
        )