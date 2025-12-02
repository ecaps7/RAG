"""Retrieval router that creates retrieval plans based on intent."""

from __future__ import annotations

from ..config import TOP_K
from ..core.types import Intent, RetrievalPlan


class RetrievalRouter:
    """Routes retrieval strategy based on intent classification."""
    
    def plan(self, intent: Intent) -> RetrievalPlan:
        """Create a retrieval plan based on the classified intent.
        
        Args:
            intent: The classified intent
            
        Returns:
            A RetrievalPlan specifying which sources to use
        """
        # Local-only intents
        if intent in {Intent.data_lookup, Intent.definition_lookup, Intent.meta_query}:
            return RetrievalPlan(
                use_local=True,
                use_web=False,
                local_top_k=TOP_K["local"],
                web_top_k=TOP_K["web"],
                hybrid_strategy="balance",
            )
        
        # Web-only intents
        elif intent in {Intent.external_context, Intent.forecast}:
            return RetrievalPlan(
                use_local=False,
                use_web=True,
                local_top_k=TOP_K["local"],
                web_top_k=TOP_K["web"],
                hybrid_strategy="balance",
            )
        
        # Hybrid (reasoning and others)
        else:
            return RetrievalPlan(
                use_local=True,
                use_web=True,
                local_top_k=TOP_K["local"],
                web_top_k=TOP_K["web"],
                hybrid_strategy="balance",
            )
