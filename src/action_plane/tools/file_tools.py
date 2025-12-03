"""File operation tools - Sensitive risk level examples.

These tools demonstrate sensitive tool implementations that require
enhanced monitoring and logging due to file system access.
"""

import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

from ...core.domain.tools import RiskLevel
from ..tool_registry import governor_tool


class ReadFileSchema(BaseModel):
    """Schema for read file tool arguments."""
    file_path: str = Field(..., description="Path to the file to read")
    max_lines: int = Field(default=100, ge=1, le=1000, description="Maximum lines to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class WriteFileSchema(BaseModel):
    """Schema for write file tool arguments."""
    file_path: str = Field(..., description="Path where to write the file")
    content: str = Field(..., description="Content to write to the file")
    encoding: str = Field(default="utf-8", description="File encoding")
    create_dirs: bool = Field(default=False, description="Create directories if they don't exist")


@governor_tool(
    name="read_file",
    description="Read content from a text file",
    risk_level=RiskLevel.SENSITIVE,
    args_schema=ReadFileSchema,
    category="file",
    tags=["file", "read", "filesystem", "sensitive"],
    timeout_seconds=30.0,
    max_retries=1,
    requires_auth=False
)
def read_file(file_path: str, max_lines: int = 100, encoding: str = "utf-8") -> dict:
    """Read content from a text file.
    
    This is a sensitive tool that accesses the file system for reading.
    It's auto-approved but requires enhanced logging.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum number of lines to read (1-1000)
        encoding: File encoding to use
        
    Returns:
        Dictionary with file content and metadata
    """
    from datetime import datetime
    
    # Security: Validate file path
    normalized_path = os.path.normpath(file_path)
    
    # Prevent directory traversal attacks
    if ".." in normalized_path or normalized_path.startswith("/"):
        return {
            "status": "error",
            "error": "Invalid file path: directory traversal not allowed",
            "file_path": file_path,
            "error_type": "SecurityError"
        }
    
    # Restrict to certain file extensions for safety
    allowed_extensions = {".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".html", ".css"}
    file_extension = Path(normalized_path).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return {
            "status": "error",
            "error": f"File type not allowed: {file_extension}",
            "file_path": file_path,
            "allowed_extensions": list(allowed_extensions),
            "error_type": "FileTypeError"
        }
    
    try:
        # Check if file exists
        if not os.path.exists(normalized_path):
            return {
                "status": "error",
                "error": "File not found",
                "file_path": normalized_path,
                "error_type": "FileNotFoundError"
            }
        
        # Get file metadata
        file_stat = os.stat(normalized_path)
        file_size = file_stat.st_size
        
        # Check file size limit (max 1MB for safety)
        max_size = 1024 * 1024  # 1MB
        if file_size > max_size:
            return {
                "status": "error",
                "error": f"File too large: {file_size} bytes (max {max_size})",
                "file_path": normalized_path,
                "file_size": file_size,
                "error_type": "FileSizeError"
            }
        
        # Read file content
        with open(normalized_path, 'r', encoding=encoding) as f:
            lines = []
            line_count = 0
            
            for line in f:
                if line_count >= max_lines:
                    break
                lines.append(line.rstrip('\n\r'))
                line_count += 1
        
        total_lines = line_count
        if line_count >= max_lines:
            # Count total lines if we hit the limit
            with open(normalized_path, 'r', encoding=encoding) as f:
                total_lines = sum(1 for _ in f)
        
        return {
            "status": "success",
            "file_path": normalized_path,
            "content": lines,
            "lines_read": len(lines),
            "total_lines": total_lines,
            "file_size": file_size,
            "encoding": encoding,
            "read_at": datetime.utcnow().isoformat(),
            "truncated": len(lines) >= max_lines
        }
        
    except UnicodeDecodeError:
        return {
            "status": "error",
            "error": f"Cannot decode file with encoding: {encoding}",
            "file_path": normalized_path,
            "encoding": encoding,
            "error_type": "UnicodeDecodeError"
        }
    except PermissionError:
        return {
            "status": "error",
            "error": "Permission denied to read file",
            "file_path": normalized_path,
            "error_type": "PermissionError"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error reading file: {str(e)}",
            "file_path": normalized_path,
            "error_type": type(e).__name__
        }


@governor_tool(
    name="write_file",
    description="Write content to a text file",
    risk_level=RiskLevel.SENSITIVE,
    args_schema=WriteFileSchema,
    category="file",
    tags=["file", "write", "filesystem", "sensitive"],
    timeout_seconds=30.0,
    max_retries=1,
    requires_auth=False
)
def write_file(
    file_path: str, 
    content: str, 
    encoding: str = "utf-8",
    create_dirs: bool = False
) -> dict:
    """Write content to a text file.
    
    This is a sensitive tool that writes to the file system.
    It requires enhanced monitoring due to potential data modification.
    
    Args:
        file_path: Path where to write the file
        content: Content to write to the file
        encoding: File encoding to use
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Dictionary with write operation results
    """
    from datetime import datetime
    
    # Security: Validate file path
    normalized_path = os.path.normpath(file_path)
    
    # Prevent directory traversal attacks
    if ".." in normalized_path or normalized_path.startswith("/"):
        return {
            "status": "error",
            "error": "Invalid file path: directory traversal not allowed",
            "file_path": file_path,
            "error_type": "SecurityError"
        }
    
    # Restrict to certain file extensions for safety
    allowed_extensions = {".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".html", ".css"}
    file_extension = Path(normalized_path).suffix.lower()
    
    if file_extension not in allowed_extensions:
        return {
            "status": "error",
            "error": f"File type not allowed for writing: {file_extension}",
            "file_path": file_path,
            "allowed_extensions": list(allowed_extensions),
            "error_type": "FileTypeError"
        }
    
    # Check content size limit (max 100KB for safety)
    max_content_size = 100 * 1024  # 100KB
    content_size = len(content.encode(encoding))
    
    if content_size > max_content_size:
        return {
            "status": "error",
            "error": f"Content too large: {content_size} bytes (max {max_content_size})",
            "content_size": content_size,
            "error_type": "ContentSizeError"
        }
    
    try:
        # Create parent directories if requested
        parent_dir = Path(normalized_path).parent
        
        if create_dirs and not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if parent directory exists
        if not parent_dir.exists():
            return {
                "status": "error",
                "error": f"Parent directory does not exist: {parent_dir}",
                "file_path": normalized_path,
                "parent_dir": str(parent_dir),
                "error_type": "DirectoryNotFoundError"
            }
        
        # Check if file already exists
        file_existed = os.path.exists(normalized_path)
        old_size = 0
        if file_existed:
            old_size = os.path.getsize(normalized_path)
        
        # Write content to file
        with open(normalized_path, 'w', encoding=encoding) as f:
            f.write(content)
        
        # Get new file info
        new_size = os.path.getsize(normalized_path)
        line_count = content.count('\n') + (1 if content else 0)
        
        return {
            "status": "success",
            "file_path": normalized_path,
            "bytes_written": new_size,
            "lines_written": line_count,
            "encoding": encoding,
            "written_at": datetime.utcnow().isoformat(),
            "file_existed": file_existed,
            "old_size": old_size if file_existed else None,
            "size_change": new_size - old_size if file_existed else new_size
        }
        
    except PermissionError:
        return {
            "status": "error",
            "error": "Permission denied to write file",
            "file_path": normalized_path,
            "error_type": "PermissionError"
        }
    except UnicodeEncodeError:
        return {
            "status": "error",
            "error": f"Cannot encode content with encoding: {encoding}",
            "file_path": normalized_path,
            "encoding": encoding,
            "error_type": "UnicodeEncodeError"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Unexpected error writing file: {str(e)}",
            "file_path": normalized_path,
            "error_type": type(e).__name__
        }