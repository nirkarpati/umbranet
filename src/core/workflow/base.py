"""Base workflow implementation using LangGraph.

This module provides the foundational workflow infrastructure
that all Governor workflows extend from.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, TypeVar

from langgraph.graph import Graph, END

from ..domain.state import GovernorState, StateNode

T = TypeVar("T", bound=GovernorState)


class WorkflowBase(ABC):
    """Abstract base class for LangGraph workflows.
    
    This class provides the common infrastructure for building
    stateful workflows using LangGraph's state machine capabilities.
    """
    
    def __init__(self) -> None:
        """Initialize the workflow base."""
        self._graph: Graph | None = None
        self._compiled_graph: Any | None = None
        
    @abstractmethod
    def build_graph(self) -> Graph:
        """Build and return the LangGraph state machine.
        
        This method must be implemented by subclasses to define
        the specific nodes, edges, and flow of their workflow.
        
        Returns:
            Configured Graph ready for compilation
        """
        pass
    
    @abstractmethod
    def get_entry_point(self) -> str:
        """Get the entry point node name for this workflow.
        
        Returns:
            Name of the node that serves as the workflow entry point
        """
        pass
    
    @abstractmethod
    def get_finish_point(self) -> str:
        """Get the finish point node name for this workflow.
        
        Returns:
            Name of the node that serves as the workflow exit point
        """
        pass
    
    def compile(self) -> Any:
        """Compile the workflow graph for execution.
        
        Returns:
            Compiled graph ready for execution
            
        Raises:
            RuntimeError: If graph compilation fails
        """
        if self._compiled_graph is not None:
            return self._compiled_graph
            
        try:
            self._graph = self.build_graph()
            self._compiled_graph = self._graph.compile()
            return self._compiled_graph
        except Exception as e:
            raise RuntimeError(f"Failed to compile workflow graph: {e}") from e
    
    def execute(
        self,
        initial_state: GovernorState,
        config: dict[str, Any] | None = None
    ) -> GovernorState:
        """Execute the workflow with the given initial state.
        
        Args:
            initial_state: Starting state for the workflow
            config: Optional configuration for execution
            
        Returns:
            Final state after workflow completion
            
        Raises:
            RuntimeError: If workflow execution fails
        """
        if self._compiled_graph is None:
            self.compile()
            
        try:
            result = self._compiled_graph.invoke(
                initial_state,
                config=config or {}
            )
            return result
        except Exception as e:
            raise RuntimeError(f"Workflow execution failed: {e}") from e

    async def aexecute(
        self,
        initial_state: GovernorState,
        config: dict[str, Any] | None = None
    ) -> GovernorState:
        """Execute the workflow asynchronously with the given initial state.
        
        Args:
            initial_state: Starting state for the workflow
            config: Optional configuration for execution
            
        Returns:
            Final state after workflow completion
            
        Raises:
            RuntimeError: If workflow execution fails
        """
        if self._compiled_graph is None:
            self.compile()
            
        try:
            # Try to use ainvoke if available, fallback to invoke
            if hasattr(self._compiled_graph, 'ainvoke'):
                result = await self._compiled_graph.ainvoke(
                    initial_state,
                    config=config or {}
                )
            else:
                # Fallback to synchronous execution
                result = self._compiled_graph.invoke(
                    initial_state,
                    config=config or {}
                )
            return result
        except Exception as e:
            raise RuntimeError(f"Workflow execution failed: {e}") from e
    
    def stream(
        self,
        initial_state: GovernorState,
        config: dict[str, Any] | None = None
    ):
        """Stream workflow execution step by step.
        
        Args:
            initial_state: Starting state for the workflow
            config: Optional configuration for execution
            
        Yields:
            State updates as workflow progresses
            
        Raises:
            RuntimeError: If workflow streaming fails
        """
        if self._compiled_graph is None:
            self.compile()
            
        try:
            for update in self._compiled_graph.stream(
                initial_state,
                config=config or {}
            ):
                yield update
        except Exception as e:
            raise RuntimeError(f"Workflow streaming failed: {e}") from e
    
    async def astream(
        self,
        initial_state: GovernorState,
        config: dict[str, Any] | None = None
    ):
        """Asynchronously stream workflow execution step by step.
        
        Args:
            initial_state: Starting state for the workflow
            config: Optional configuration for execution
            
        Yields:
            State updates as workflow progresses
            
        Raises:
            RuntimeError: If workflow streaming fails
        """
        if self._compiled_graph is None:
            self.compile()
            
        try:
            async for update in self._compiled_graph.astream(
                initial_state,
                config=config or {}
            ):
                yield update
        except Exception as e:
            raise RuntimeError(f"Workflow streaming failed: {e}") from e
    
    def get_graph_visualization(self) -> str:
        """Get a visual representation of the workflow graph.
        
        Returns:
            Mermaid diagram representation of the workflow
            
        Raises:
            RuntimeError: If graph is not compiled
        """
        if self._compiled_graph is None:
            raise RuntimeError("Graph must be compiled before visualization")
            
        try:
            return self._compiled_graph.get_graph().draw_mermaid()
        except Exception as e:
            raise RuntimeError(f"Failed to generate graph visualization: {e}") from e
    
    def get_state_transitions(self) -> dict[str, list[str]]:
        """Get mapping of possible state transitions.
        
        Returns:
            Dictionary mapping node names to their possible next nodes
            
        Raises:
            RuntimeError: If graph is not compiled
        """
        if self._compiled_graph is None:
            raise RuntimeError("Graph must be compiled before getting transitions")
            
        # Extract transition information from the compiled graph
        graph_data = self._compiled_graph.get_graph()
        transitions = {}
        
        for node in graph_data.nodes:
            transitions[node] = []
            
        for edge in graph_data.edges:
            source = edge.source
            target = edge.target
            if source not in transitions:
                transitions[source] = []
            transitions[source].append(target)
            
        return transitions


class NodeFunction(ABC):
    """Abstract base class for workflow node functions.
    
    Node functions implement the business logic for individual
    steps in the workflow state machine.
    """
    
    @abstractmethod
    def __call__(self, state: GovernorState) -> GovernorState:
        """Execute the node function.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated state after node execution
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the node name for registration with LangGraph."""
        pass


class ConditionalFunction(ABC):
    """Abstract base class for conditional routing functions.
    
    Conditional functions determine the next node in the workflow
    based on current state conditions.
    """
    
    @abstractmethod
    def __call__(self, state: GovernorState) -> str:
        """Determine the next node based on state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Name of the next node to execute
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the conditional function name."""
        pass