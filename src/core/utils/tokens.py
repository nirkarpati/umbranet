"""Token counting utilities for managing context window limits."""


import tiktoken

from ..domain.memory import ContextObject, ConversationTurn


class TokenCounter:
    """Utility for counting tokens in text using tiktoken."""
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize token counter.
        
        Args:
            encoding_name: Name of the tiktoken encoding to use.
                          "cl100k_base" is used by GPT-4, GPT-3.5-turbo
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))
    
    def count_conversation_turn_tokens(self, turn: ConversationTurn) -> int:
        """Count tokens in a conversation turn (user + assistant messages)."""
        user_tokens = self.count_tokens(turn.user_message)
        assistant_tokens = self.count_tokens(turn.assistant_response)
        # Add small overhead for message formatting
        return user_tokens + assistant_tokens + 10
    
    def count_context_tokens(self, context: ContextObject) -> int:
        """Count total tokens in a context object."""
        summary_tokens = self.count_tokens(context.summary or "")
        messages_tokens = sum(
            self.count_conversation_turn_tokens(turn) 
            for turn in context.recent_messages
        )
        return summary_tokens + messages_tokens
    
    def estimate_turns_within_budget(
        self, 
        turns: list[ConversationTurn], 
        token_budget: int,
        summary_tokens: int = 0
    ) -> tuple[list[ConversationTurn], int]:
        """Find how many recent turns fit within token budget.
        
        Args:
            turns: List of conversation turns (most recent first)
            token_budget: Maximum tokens allowed
            summary_tokens: Tokens already used by summary
        
        Returns:
            Tuple of (turns_that_fit, actual_token_count)
        """
        if not turns:
            return [], summary_tokens
        
        current_tokens = summary_tokens
        turns_that_fit = []
        
        for turn in turns:
            turn_tokens = self.count_conversation_turn_tokens(turn)
            if current_tokens + turn_tokens > token_budget:
                break
            
            turns_that_fit.append(turn)
            current_tokens += turn_tokens
        
        return turns_that_fit, current_tokens


# Global token counter instance
token_counter = TokenCounter()