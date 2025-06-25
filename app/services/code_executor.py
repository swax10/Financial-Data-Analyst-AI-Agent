"""
Code Execution Service for Financial Data Analysis AI Agent
"""

import asyncio
import logging
import json
import uuid
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path
import subprocess
import sys
import io
import contextlib
from concurrent.futures import ThreadPoolExecutor

from app.models.schemas import ExecutionResult, OutputMessage, Result, ExecutionError
from app.config import settings

logger = logging.getLogger(__name__)


class CodeExecutor:
    """
    Code execution service with secure Python environment
    with security and sandboxing considerations
    """

    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.global_namespace = self._create_safe_namespace()

    async def initialize(self):
        """Initialize the code execution environment"""
        logger.info("Initializing code execution environment...")

        # Pre-import common libraries
        await self._preload_libraries()

        logger.info("Code execution environment ready")

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up code executor...")
        self.executor.shutdown(wait=True)

    def _create_safe_namespace(self) -> Dict[str, Any]:
        """Create a safe execution namespace with allowed imports"""
        import builtins

        # Create a safer version of builtins
        safe_builtins = {}

        # Allow most safe built-ins
        safe_builtin_names = [
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
            'min', 'max', 'sum', 'abs', 'round', 'sorted', 'reversed',
            'enumerate', 'zip', 'range', 'print', 'type', 'isinstance',
            'hasattr', 'getattr', 'setattr', 'delattr', 'callable',
            'iter', 'next', 'map', 'filter', 'any', 'all',
            'ord', 'chr', 'bin', 'hex', 'oct', 'pow', 'divmod',
            '__import__'  # Allow import for pandas, numpy etc.
        ]

        for name in safe_builtin_names:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        namespace = {
            '__builtins__': safe_builtins
        }
        return namespace

    async def _preload_libraries(self):
        """Preload common libraries for faster execution"""
        preload_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for non-interactive use
plt.switch_backend('Agg')
"""

        try:
            await self._execute_code_safe(preload_code, "system")

            # Add multi-sheet analysis function to global namespace
            self.global_namespace['analyze_multiple_sheets'] = self._analyze_multiple_sheets

            logger.info("Libraries preloaded successfully")
        except Exception as e:
            logger.warning(f"Failed to preload libraries: {str(e)}")

    async def execute_code(self, code: str, session_id: str,
                          timeout: Optional[int] = None) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment
        Secure code execution with output capture
        """
        if timeout is None:
            timeout = settings.EXECUTION_TIMEOUT

        start_time = time.time()
        execution_count = self._get_execution_count(session_id)

        try:
            logger.info(f"Executing code for session {session_id}")
            logger.info(f"Code length: {len(code)} characters")
            logger.info(f"Code preview (first 300 chars):\n{code[:300]}")
            logger.info(f"Full code being executed:\n{code}")

            # Validate code safety
            self._validate_code_safety(code)

            # Execute code
            result = await asyncio.wait_for(
                self._execute_code_safe(code, session_id),
                timeout=timeout
            )

            execution_time = time.time() - start_time

            logger.info(f"=== CODE EXECUTION RESULT ===")
            logger.info(f"Execution time: {execution_time:.2f}s")
            logger.info(f"Results count: {len(result.get('results', []))}")
            logger.info(f"Stdout lines: {len(result.get('logs', {}).get('stdout', []))}")
            logger.info(f"Stderr lines: {len(result.get('logs', {}).get('stderr', []))}")

            if result.get('results'):
                logger.info(f"First result preview: {str(result['results'][0])[:200]}")

            return ExecutionResult(
                success=True,
                results=result.get('results', []),
                logs=result.get('logs', {"stdout": [], "stderr": []}),
                execution_count=execution_count,
                execution_time=execution_time
            )

        except asyncio.TimeoutError:
            logger.error(f"Code execution timeout for session {session_id}")
            return ExecutionResult(
                success=False,
                error={
                    "name": "TimeoutError",
                    "value": f"Execution timeout after {timeout} seconds",
                    "traceback": ""
                },
                execution_count=execution_count,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            return ExecutionResult(
                success=False,
                error={
                    "name": type(e).__name__,
                    "value": str(e),
                    "traceback": self._format_traceback(e)
                },
                execution_count=execution_count,
                execution_time=time.time() - start_time
            )

    async def _execute_code_safe(self, code: str, session_id: str) -> Dict[str, Any]:
        """Execute code in a safe environment"""

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        results = []
        logs = {"stdout": [], "stderr": []}

        # Get or create session namespace
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "namespace": self.global_namespace.copy(),
                "execution_count": 0
            }

        session_namespace = self.active_sessions[session_id]["namespace"]

        try:
            # Execute in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._execute_in_thread,
                code,
                session_namespace,
                stdout_capture,
                stderr_capture
            )

            # Capture outputs
            stdout_content = stdout_capture.getvalue()
            stderr_content = stderr_capture.getvalue()

            if stdout_content:
                logs["stdout"] = stdout_content.strip().split('\n')

            if stderr_content:
                logs["stderr"] = stderr_content.strip().split('\n')

            # Process results
            if result is not None:
                results.append(self._format_result(result))

            # Update execution count
            self.active_sessions[session_id]["execution_count"] += 1

            return {
                "results": results,
                "logs": logs
            }

        except Exception as e:
            raise e
        finally:
            stdout_capture.close()
            stderr_capture.close()

    def _execute_in_thread(self, code: str, namespace: Dict[str, Any],
                          stdout_capture: io.StringIO, stderr_capture: io.StringIO):
        """Execute code in a separate thread"""

        with contextlib.redirect_stdout(stdout_capture), \
             contextlib.redirect_stderr(stderr_capture):

            try:
                # Compile and execute
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, namespace)

                # Try to get the last expression result
                lines = code.strip().split('\n')
                if lines:
                    last_line = lines[-1].strip()
                    if last_line and not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'with ')):
                        try:
                            result = eval(last_line, namespace)
                            return result
                        except:
                            pass

                return None

            except Exception as e:
                raise e

    def _validate_code_safety(self, code: str):
        """Validate code for security issues"""

        # Forbidden patterns
        forbidden_patterns = [
            'import os', 'import sys', 'import subprocess',
            'exec(', 'eval(', '__import__',
            'open(', 'file(', 'input(',
            'raw_input(', 'compile(',
            'globals()', 'locals()', 'vars()',
            'dir()', 'help(', 'exit(', 'quit(',
        ]

        code_lower = code.lower()
        for pattern in forbidden_patterns:
            if pattern in code_lower:
                # Allow some safe patterns
                if pattern == 'open(' and 'workspace/' in code:
                    continue  # Allow opening files in workspace
                raise ValueError(f"Forbidden operation detected: {pattern}")

        # Check imports against allowed list
        import ast
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in settings.ALLOWED_IMPORTS:
                            raise ValueError(f"Import not allowed: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in settings.ALLOWED_IMPORTS:
                        raise ValueError(f"Import not allowed: {node.module}")
        except SyntaxError:
            raise ValueError("Invalid Python syntax")

    def _format_result(self, result: Any) -> Result:
        """Format execution result for API response"""

        if result is None:
            return Result()

        # Handle different result types
        if hasattr(result, '_repr_html_'):
            return Result(html=result._repr_html_(), is_main_result=True)

        elif hasattr(result, '_repr_svg_'):
            return Result(svg=result._repr_svg_(), is_main_result=True)

        elif hasattr(result, '_repr_png_'):
            return Result(png=result._repr_png_(), is_main_result=True)

        elif hasattr(result, '_repr_json_'):
            return Result(json=result._repr_json_(), is_main_result=True)

        else:
            # Default text representation
            text_repr = str(result)
            return Result(text=text_repr, is_main_result=True)

    def _format_traceback(self, exception: Exception) -> str:
        """Format exception traceback"""
        import traceback
        return traceback.format_exc()

    def _get_execution_count(self, session_id: str) -> int:
        """Get execution count for session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]["execution_count"] + 1
        return 1

    async def stream_execution(self, code: str, session_id: str) -> AsyncGenerator[OutputMessage, None]:
        """Stream execution output in real-time"""

        # This is a simplified streaming implementation
        # In a full implementation, you'd use Jupyter kernel's messaging

        start_time = int(time.time() * 1000)

        yield OutputMessage(
            line="Starting execution...",
            timestamp=start_time,
            error=False
        )

        try:
            result = await self.execute_code(code, session_id)

            # Stream stdout
            for line in result.logs.get("stdout", []):
                yield OutputMessage(
                    line=line,
                    timestamp=int(time.time() * 1000),
                    error=False
                )

            # Stream stderr
            for line in result.logs.get("stderr", []):
                yield OutputMessage(
                    line=line,
                    timestamp=int(time.time() * 1000),
                    error=True
                )

            # Stream results
            for result_item in result.results:
                if result_item.text:
                    yield OutputMessage(
                        line=result_item.text,
                        timestamp=int(time.time() * 1000),
                        error=False
                    )

            yield OutputMessage(
                line="Execution completed successfully",
                timestamp=int(time.time() * 1000),
                error=False
            )

        except Exception as e:
            yield OutputMessage(
                line=f"Execution failed: {str(e)}",
                timestamp=int(time.time() * 1000),
                error=True
            )

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        return self.active_sessions.get(session_id)

    def cleanup_session(self, session_id: str):
        """Cleanup session data"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Session {session_id} cleaned up")

    def _analyze_multiple_sheets(self, session_id: str, analysis_type: str = "join",
                               sheets_to_use: List[str] = None, join_columns: List[str] = None,
                               metrics: List[str] = None) -> Dict[str, Any]:
        """
        Multi-sheet analysis function available to LLM-generated code
        This function provides SQL-like JOIN capabilities for Excel sheets
        """
        import pandas as pd

        # Get session data
        session_data = self.active_sessions.get(session_id)
        if not session_data:
            return {"error": "Session not found", "success": False}

        # Get all sheets data from data processor
        data_processor = session_data.get('data_processor')
        if not data_processor or not hasattr(data_processor, '_all_sheets_data'):
            return {"error": "No multi-sheet data available", "success": False}

        all_sheets_data = data_processor._all_sheets_data
        sheet_metadata = data_processor._sheet_metadata

        if not all_sheets_data:
            return {"error": "No sheets data found", "success": False}

        # Default to all sheets if none specified
        if not sheets_to_use:
            sheets_to_use = list(all_sheets_data.keys())

        # Validate sheet names
        invalid_sheets = [s for s in sheets_to_use if s not in all_sheets_data]
        if invalid_sheets:
            return {
                "error": f"Invalid sheet names: {invalid_sheets}. Available: {list(all_sheets_data.keys())}",
                "success": False
            }

        try:
            result = {"success": True, "analysis_type": analysis_type}

            if analysis_type == "join":
                result.update(self._perform_sheet_join(all_sheets_data, sheets_to_use, join_columns, sheet_metadata))

            elif analysis_type == "compare":
                result.update(self._perform_sheet_comparison(all_sheets_data, sheets_to_use, metrics))

            elif analysis_type == "aggregate":
                result.update(self._perform_sheet_aggregation(all_sheets_data, sheets_to_use, metrics))

            elif analysis_type == "relationship":
                result.update(self._analyze_sheet_relationships_runtime(all_sheets_data, sheets_to_use))

            else:
                return {"error": f"Unknown analysis type: {analysis_type}", "success": False}

            return result

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "success": False}

    def _perform_sheet_join(self, all_sheets_data: Dict, sheets_to_use: List[str],
                          join_columns: List[str], sheet_metadata: Dict) -> Dict[str, Any]:
        """Perform SQL-style JOIN between sheets"""
        import pandas as pd

        if len(sheets_to_use) < 2:
            return {"error": "Need at least 2 sheets for join", "joined_data": None}

        # Start with first sheet
        result_df = all_sheets_data[sheets_to_use[0]].copy()
        join_info = [f"Started with sheet '{sheets_to_use[0]}' ({len(result_df)} rows)"]

        # Join with subsequent sheets
        for sheet_name in sheets_to_use[1:]:
            sheet_df = all_sheets_data[sheet_name]

            # Determine join columns
            if join_columns:
                # Use specified columns
                join_cols = [col for col in join_columns if col in result_df.columns and col in sheet_df.columns]
            else:
                # Auto-detect join columns
                join_cols = list(set(result_df.columns) & set(sheet_df.columns))

            if not join_cols:
                join_info.append(f"No common columns found with '{sheet_name}', skipping")
                continue

            # Perform inner join
            before_count = len(result_df)
            result_df = pd.merge(result_df, sheet_df, on=join_cols, how='inner', suffixes=('', f'_{sheet_name}'))
            after_count = len(result_df)

            join_info.append(f"Joined with '{sheet_name}' on {join_cols}: {before_count} → {after_count} rows")

        return {
            "joined_data": result_df,
            "join_summary": join_info,
            "final_shape": result_df.shape,
            "final_columns": list(result_df.columns)
        }

    def _perform_sheet_comparison(self, all_sheets_data: Dict, sheets_to_use: List[str],
                                metrics: List[str]) -> Dict[str, Any]:
        """Compare metrics across sheets"""
        import pandas as pd

        comparison_results = {}

        for sheet_name in sheets_to_use:
            df = all_sheets_data[sheet_name]
            sheet_metrics = {}

            # Calculate basic metrics for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns

            for col in numeric_cols:
                if metrics and col not in metrics:
                    continue

                sheet_metrics[col] = {
                    'sum': float(df[col].sum()),
                    'mean': float(df[col].mean()),
                    'count': int(df[col].count()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }

            comparison_results[sheet_name] = sheet_metrics

        return {
            "comparison_data": comparison_results,
            "sheets_compared": sheets_to_use
        }

    def _perform_sheet_aggregation(self, all_sheets_data: Dict, sheets_to_use: List[str],
                                 metrics: List[str]) -> Dict[str, Any]:
        """Aggregate data across multiple sheets"""
        import pandas as pd

        # Combine all specified sheets
        combined_dfs = []
        for sheet_name in sheets_to_use:
            df = all_sheets_data[sheet_name].copy()
            df['_source_sheet'] = sheet_name
            combined_dfs.append(df)

        combined_df = pd.concat(combined_dfs, ignore_index=True)

        # Calculate aggregations
        aggregations = {}
        numeric_cols = combined_df.select_dtypes(include=['number']).columns

        for col in numeric_cols:
            if metrics and col not in metrics:
                continue

            aggregations[col] = {
                'total_sum': float(combined_df[col].sum()),
                'overall_mean': float(combined_df[col].mean()),
                'total_count': int(combined_df[col].count()),
                'by_sheet': combined_df.groupby('_source_sheet')[col].sum().to_dict()
            }

        return {
            "aggregated_data": aggregations,
            "combined_shape": combined_df.shape,
            "source_sheets": sheets_to_use
        }

    def _analyze_sheet_relationships_runtime(self, all_sheets_data: Dict,
                                           sheets_to_use: List[str]) -> Dict[str, Any]:
        """Analyze relationships between sheets at runtime"""

        relationships = {}

        for i, sheet1 in enumerate(sheets_to_use):
            for sheet2 in sheets_to_use[i+1:]:
                df1 = all_sheets_data[sheet1]
                df2 = all_sheets_data[sheet2]

                # Find common columns
                common_cols = list(set(df1.columns) & set(df2.columns))

                if common_cols:
                    relationships[f"{sheet1}__{sheet2}"] = {
                        'common_columns': common_cols,
                        'sheet1_shape': df1.shape,
                        'sheet2_shape': df2.shape,
                        'potential_joins': common_cols
                    }

        return {
            "relationships": relationships,
            "sheets_analyzed": sheets_to_use
        }
