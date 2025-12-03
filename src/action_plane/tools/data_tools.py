"""Data processing tools - Safe/Sensitive risk level examples.

These tools demonstrate safe and sensitive tool implementations
for data processing and calculations.
"""

import ast
import math
import operator
import os
import re
import requests
from typing import List, Dict, Any

from pydantic import BaseModel, Field

from ...core.domain.tools import RiskLevel
from ..tool_registry import governor_tool


class SearchSchema(BaseModel):
    """Schema for web search tool arguments."""
    query: str = Field(..., description="Search query string")
    num_results: int = Field(default=5, ge=1, le=20, description="Number of results to return (1-20)")
    safe_search: bool = Field(default=True, description="Enable safe search filtering")


class CalculateSchema(BaseModel):
    """Schema for calculator tool arguments."""
    expression: str = Field(..., description="Mathematical expression to evaluate")
    precision: int = Field(default=2, ge=0, le=10, description="Decimal precision for results")


@governor_tool(
    name="search_web",
    description="Search the web for information on a given query",
    risk_level=RiskLevel.SAFE,
    args_schema=SearchSchema,
    category="data",
    tags=["search", "web", "information", "safe"],
    timeout_seconds=20.0,
    max_retries=2
)
def search_web(query: str, num_results: int = 5, safe_search: bool = True) -> dict:
    """Search the web for information using SerpAPI.
    
    This is a safe tool that provides web search results without
    security risks or side effects.
    
    Args:
        query: Search query string
        num_results: Number of results to return (1-20)
        safe_search: Enable safe search filtering
        
    Returns:
        Dictionary with search results
    """
    import uuid
    from datetime import datetime
    
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return {
            "status": "error",
            "error": "SerpAPI key not configured",
            "message": "Set SERPAPI_KEY environment variable"
        }
    
    try:
        # Use SerpAPI for Google search
        url = "https://serpapi.com/search"
        params = {
            "engine": "google",
            "q": query,
            "num": min(num_results, 20),
            "api_key": serpapi_key,
            "safe": "active" if safe_search else "off"
        }
        
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        organic_results = data.get("organic_results", [])
        
        for i, result in enumerate(organic_results[:num_results]):
            results.append({
                "rank": i + 1,
                "title": result.get("title", "No title"),
                "url": result.get("link", ""),
                "snippet": result.get("snippet", "No snippet available"),
                "displayed_link": result.get("displayed_link", ""),
                "search_id": str(uuid.uuid4())
            })
        
        return {
            "status": "success",
            "query": query,
            "num_results": len(results),
            "results": results,
            "search_time_ms": data.get("search_metadata", {}).get("total_time_taken", 0) * 1000,
            "safe_search": safe_search,
            "search_timestamp": datetime.utcnow().isoformat(),
            "source": "serpapi_google_search",
            "search_id": data.get("search_metadata", {}).get("id"),
            "total_found": data.get("search_information", {}).get("total_results", "Unknown")
        }
        
    except requests.exceptions.RequestException as e:
        return {
            "status": "error",
            "error": "Search API request failed",
            "message": str(e)
        }
    except (KeyError, ValueError) as e:
        return {
            "status": "error",
            "error": "Invalid search API response",
            "message": str(e)
        }


@governor_tool(
    name="calculate",
    description="Perform mathematical calculations and evaluate expressions",
    risk_level=RiskLevel.SAFE,
    args_schema=CalculateSchema,
    category="utility",
    tags=["math", "calculator", "computation", "safe"],
    timeout_seconds=5.0,
    max_retries=1
)
def calculate(expression: str, precision: int = 2) -> dict:
    """Perform mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate
        precision: Decimal precision for results (0-10)
        
    Returns:
        Dictionary with calculation results
    """
    import ast
    import operator
    
    # Safe mathematical operators
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }
    
    # Safe mathematical functions
    safe_functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def safe_eval(node):
        """Safely evaluate mathematical expressions."""
        if isinstance(node, ast.Expression):
            return safe_eval(node.body)
        elif isinstance(node, ast.Constant):  # Numbers
            return node.value
        elif isinstance(node, ast.Name):  # Variables/constants
            if node.id in safe_functions:
                return safe_functions[node.id]
            else:
                raise ValueError(f"Unsafe name: {node.id}")
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = safe_eval(node.left)
            right = safe_eval(node.right)
            op = safe_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsafe operator: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = safe_eval(node.operand)
            op = safe_operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsafe unary operator: {type(node.op)}")
            return op(operand)
        elif isinstance(node, ast.Call):  # Function calls
            func = safe_eval(node.func)
            args = [safe_eval(arg) for arg in node.args]
            if callable(func):
                return func(*args)
            else:
                raise ValueError(f"Not a callable: {func}")
        else:
            raise ValueError(f"Unsafe operation: {type(node)}")
    
    try:
        # Clean the expression
        expression = expression.strip()
        
        # Replace common constants
        expression = expression.replace("π", "pi").replace("π", "pi")
        
        # Parse and evaluate
        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree)
        
        # Apply precision
        if isinstance(result, float):
            result = round(result, precision)
        
        return {
            "status": "success",
            "expression": expression,
            "result": result,
            "result_type": type(result).__name__,
            "precision": precision
        }
        
    except ZeroDivisionError:
        return {
            "status": "error",
            "expression": expression,
            "error": "Division by zero",
            "error_type": "ZeroDivisionError"
        }
    except ValueError as e:
        return {
            "status": "error",
            "expression": expression,
            "error": str(e),
            "error_type": "ValueError"
        }
    except Exception as e:
        return {
            "status": "error",
            "expression": expression,
            "error": f"Calculation error: {str(e)}",
            "error_type": type(e).__name__
        }