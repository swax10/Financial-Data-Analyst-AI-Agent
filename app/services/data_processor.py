"""
Data Processing Service for Financial Data Analysis AI Agent
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import chardet
import json
from datetime import datetime
from fastapi import HTTPException

from app.models.schemas import DataSummary, FinancialPattern
from app.config import settings
from .column_mapper import ColumnMapper
from app.utils.file_errors import FileError, FileErrorType, create_user_friendly_error

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processing service that handles various file formats
    and provides intelligent analysis for financial data
    """

    def __init__(self):
        self.supported_formats = {
            '.csv': self._read_csv,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel,
            '.json': self._read_json,
            '.parquet': self._read_parquet
        }
        self.column_mapper = ColumnMapper()

    async def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze uploaded file and extract metadata
        Comprehensive file analysis with metadata extraction
        """
        try:
            logger.info(f"Analyzing file: {file_path}")

            # Get file info
            file_size = file_path.stat().st_size
            file_extension = file_path.suffix.lower()

            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")

            # Read and analyze data
            df, encoding = await self._read_file(file_path)

            # Generate data summary
            data_summary = self._generate_data_summary(df, file_path, file_size, encoding)

            # Detect financial patterns
            financial_patterns = self._detect_financial_patterns(df)

            # Analyze column mapping
            column_analysis = self.column_mapper.analyze_columns(df)

            # Include sheet information if available (for Excel files)
            result = {
                "file_type": file_extension,
                "file_size": file_size,
                "encoding": encoding,
                "columns": data_summary.columns,
                "column_types": data_summary.column_types,
                "row_count": data_summary.row_count,
                "summary": data_summary.dict(),
                "financial_patterns": financial_patterns.dict(),
                "sample_data": data_summary.sample_data,
                "column_mapping": column_analysis
            }

            # Add ALL sheets metadata for multi-sheet analysis
            if hasattr(self, '_all_sheets_data') and self._all_sheets_data:
                result["all_sheets"] = {
                    "sheet_count": len(self._all_sheets_data),
                    "sheet_names": list(self._all_sheets_data.keys()),
                    "metadata": self._sheet_metadata,
                    "primary_sheet": self._select_primary_sheet(self._all_sheets_data, self._sheet_metadata)
                }

                # Add relationship analysis between sheets
                result["all_sheets"]["relationships"] = self._analyze_sheet_relationships()

                # DEEP DEBUG: Log the final result structure
                logger.info("🎯" * 80)
                logger.info("🎯 FINAL RESULT ASSEMBLY - DEEP DEBUG")
                logger.info("🎯" * 80)
                logger.info(f"🎯 Result keys: {list(result.keys())}")
                logger.info(f"🎯 all_sheets keys: {list(result['all_sheets'].keys())}")
                logger.info(f"🎯 Sheet count in result: {result['all_sheets']['sheet_count']}")
                logger.info(f"🎯 Sheet names in result: {result['all_sheets']['sheet_names']}")
                logger.info(f"🎯 Primary sheet in result: {result['all_sheets']['primary_sheet']}")
                logger.info(f"🎯 Metadata keys in result: {list(result['all_sheets']['metadata'].keys())}")
                logger.info("🎯" * 80)
            else:
                logger.info("🎯 No multi-sheet data found - single sheet file")

            return result

        except Exception as e:
            logger.error(f"File analysis failed: {str(e)}")
            raise

    async def _read_file(self, file_path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
        """Read file based on extension with encoding detection"""
        file_extension = file_path.suffix.lower()
        encoding = None

        # Detect encoding for text-based files
        if file_extension in ['.csv', '.json']:
            encoding = self._detect_encoding(file_path)

        # Read file using appropriate method
        reader_func = self.supported_formats[file_extension]
        df = reader_func(file_path, encoding)

        return df, encoding

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding"""
        if not settings.ENCODING_DETECTION:
            return 'utf-8'

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'

    def _read_csv(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """Read CSV file with intelligent parsing and detailed error handling"""
        file_name = file_path.name

        try:
            # Try different separators and configurations
            separators = [',', ';', '\t', '|']

            for sep in separators:
                try:
                    df = pd.read_csv(
                        file_path,
                        encoding=encoding,
                        sep=sep,
                        nrows=5  # Test with first 5 rows
                    )
                    if len(df.columns) > 1:  # Found good separator
                        # Read full file
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            sep=sep,
                            low_memory=False
                        )
                        return df
                except Exception:
                    continue

            # Fallback to default
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)

        except UnicodeDecodeError as e:
            FileError.raise_http_exception(
                FileErrorType.ENCODING_ERROR,
                additional_info=f"Failed to decode with {encoding} encoding. Try saving as UTF-8.",
                file_name=file_name
            )
        except pd.errors.EmptyDataError:
            FileError.raise_http_exception(
                FileErrorType.EMPTY_FILE,
                additional_info="CSV file contains no data",
                file_name=file_name
            )
        except pd.errors.ParserError as e:
            FileError.raise_http_exception(
                FileErrorType.CORRUPTED_FILE,
                additional_info=f"CSV parsing failed: {str(e)}",
                file_name=file_name
            )
        except Exception as e:
            logger.error(f"CSV reading failed: {str(e)}")
            error_response = create_user_friendly_error(e, file_name)
            raise HTTPException(status_code=400, detail=error_response)

    def _read_excel(self, file_path: Path, encoding: str = None) -> pd.DataFrame:
        """Read Excel file with ALL sheets support for multi-table analysis"""
        file_name = file_path.name

        try:
            # First, get all sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            logger.info(f"Excel file has {len(sheet_names)} sheets: {sheet_names}")

            # NEW APPROACH: Load ALL sheets and create metadata
            all_sheets_data = {}
            sheet_metadata = {}

            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)

                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' is empty, skipping")
                        continue

                    # Store the actual data
                    all_sheets_data[sheet_name] = df

                    # Generate metadata for this sheet
                    sheet_metadata[sheet_name] = self._generate_sheet_metadata(df, sheet_name)

                    logger.info(f"Loaded sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")

                except Exception as e:
                    logger.warning(f"Could not read sheet '{sheet_name}': {e}")
                    continue

            if not all_sheets_data:
                FileError.raise_http_exception(
                    FileErrorType.EMPTY_FILE,
                    additional_info=f"No readable data found in any of the {len(sheet_names)} sheets",
                    file_name=file_name
                )

            # Store ALL sheets data and metadata for LLM access
            self._all_sheets_data = all_sheets_data
            self._sheet_metadata = sheet_metadata

            # DEEP DEBUG: Log the metadata being created
            logger.info("📊" * 80)
            logger.info("📊 MULTI-SHEET METADATA GENERATION - DEEP DEBUG")
            logger.info("📊" * 80)
            logger.info(f"📊 Total sheets loaded: {len(all_sheets_data)}")
            logger.info(f"📊 Sheet names: {list(all_sheets_data.keys())}")
            logger.info(f"📊 Metadata keys: {list(sheet_metadata.keys())}")

            for sheet_name, metadata in sheet_metadata.items():
                logger.info(f"📊 Sheet '{sheet_name}':")
                logger.info(f"📊   - Rows: {metadata.get('row_count', 0)}")
                logger.info(f"📊   - Columns: {metadata.get('columns', [])}")
                logger.info(f"📊   - Financial score: {metadata.get('financial_score', 0):.1f}/10.0")
                logger.info(f"📊   - Description: {metadata.get('description', 'N/A')}")

            logger.info("📊" * 80)

            # For backward compatibility, return the "primary" sheet
            # But now LLM can access all sheets via metadata
            primary_sheet = self._select_primary_sheet(all_sheets_data, sheet_metadata)

            logger.info(f"Loaded {len(all_sheets_data)} sheets, primary: '{primary_sheet}'")

            return all_sheets_data[primary_sheet]

        except Exception as e:
            error_str = str(e).lower()

            # Check for specific Excel errors
            if 'password' in error_str or 'encrypted' in error_str:
                FileError.raise_http_exception(
                    FileErrorType.PASSWORD_PROTECTED,
                    additional_info="Excel file is password protected",
                    file_name=file_name
                )
            elif 'corrupted' in error_str or 'invalid' in error_str or 'not a valid' in error_str:
                FileError.raise_http_exception(
                    FileErrorType.CORRUPTED_FILE,
                    additional_info=f"Excel file appears to be corrupted: {str(e)}",
                    file_name=file_name
                )
            elif 'xlrd' in error_str or 'openpyxl' in error_str:
                FileError.raise_http_exception(
                    FileErrorType.UNSUPPORTED_FORMAT,
                    additional_info="Excel file format not supported. Try saving as .xlsx",
                    file_name=file_name
                )
            else:
                logger.error(f"Excel reading failed: {str(e)}")
                error_response = create_user_friendly_error(e, file_name)
                raise HTTPException(status_code=400, detail=error_response)

    def _read_json(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """Read JSON file"""
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError("Unsupported JSON structure")

            return df
        except Exception as e:
            logger.error(f"JSON reading failed: {str(e)}")
            raise

    def _read_parquet(self, file_path: Path, encoding: str = None) -> pd.DataFrame:
        """Read Parquet file"""
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            logger.error(f"Parquet reading failed: {str(e)}")
            raise

    def _generate_data_summary(self, df: pd.DataFrame, file_path: Path,
                             file_size: int, encoding: str) -> DataSummary:
        """Generate comprehensive data summary"""

        # Basic info
        columns = df.columns.tolist()
        column_types = {col: str(df[col].dtype) for col in columns}
        row_count = len(df)

        # Detect column categories
        date_columns = self._detect_date_columns(df)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # Missing values
        missing_values = df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items() if v > 0}

        # Sample data (first 5 rows)
        sample_data = df.head(5).fillna('').to_dict('records')

        # Clean sample data for JSON serialization
        for row in sample_data:
            for key, value in row.items():
                if pd.isna(value) or value is pd.NaT:
                    row[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    row[key] = float(value) if np.isfinite(value) else None

        return DataSummary(
            file_type=file_path.suffix.lower(),
            encoding=encoding,
            columns=columns,
            column_types=column_types,
            row_count=row_count,
            file_size=file_size,
            has_header=True,  # Assume header for now
            date_columns=date_columns,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            missing_values=missing_values,
            sample_data=sample_data
        )

    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect date columns in the dataframe"""
        date_columns = []

        for col in df.columns:
            # Check column name patterns
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['date', 'time', 'created', 'updated', 'timestamp']):
                date_columns.append(col)
                continue

            # Try to parse as date
            try:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    date_columns.append(col)
            except Exception:
                pass

        return date_columns

    def _detect_financial_patterns(self, df: pd.DataFrame) -> FinancialPattern:
        """Detect financial patterns in the data"""

        expense_columns = []
        income_columns = []
        amount_columns = []
        category_columns = []
        balance_columns = []

        for col in df.columns:
            col_lower = col.lower()

            # Expense patterns
            if any(keyword in col_lower for keyword in ['expense', 'cost', 'spend', 'payment', 'debit']):
                expense_columns.append(col)

            # Income patterns
            elif any(keyword in col_lower for keyword in ['income', 'revenue', 'salary', 'credit', 'deposit']):
                income_columns.append(col)

            # Amount patterns
            elif any(keyword in col_lower for keyword in ['amount', 'value', 'price', 'total', 'sum']):
                amount_columns.append(col)

            # Category patterns
            elif any(keyword in col_lower for keyword in ['category', 'type', 'class', 'group']):
                category_columns.append(col)

            # Balance patterns
            elif any(keyword in col_lower for keyword in ['balance', 'remaining', 'available']):
                balance_columns.append(col)

        # Calculate confidence score
        total_columns = len(df.columns)
        financial_columns = len(set(expense_columns + income_columns + amount_columns +
                                  category_columns + balance_columns))
        confidence_score = financial_columns / total_columns if total_columns > 0 else 0.0

        return FinancialPattern(
            expense_columns=expense_columns,
            income_columns=income_columns,
            date_columns=self._detect_date_columns(df),
            category_columns=category_columns,
            amount_columns=amount_columns,
            balance_columns=balance_columns,
            confidence_score=confidence_score
        )

    def _score_sheet_for_financial_data(self, df: pd.DataFrame, sheet_name: str) -> float:
        """
        Score a sheet based on how likely it contains financial data
        Returns a score from 0.0 to 10.0
        """
        if df.empty or len(df.columns) == 0:
            return 0.0

        score = 0.0

        # 1. Sheet name scoring (2 points max)
        sheet_name_lower = sheet_name.lower()
        financial_sheet_keywords = [
            'expense', 'revenue', 'income', 'cost', 'budget', 'financial',
            'transaction', 'payment', 'sales', 'profit', 'cash', 'money',
            'data', 'summary', 'report', 'analysis', 'detail'
        ]

        for keyword in financial_sheet_keywords:
            if keyword in sheet_name_lower:
                score += 0.5
                break

        # Bonus for common financial sheet names
        if any(name in sheet_name_lower for name in ['sheet1', 'data', 'main', 'summary']):
            score += 0.5

        # 2. Column name scoring (3 points max)
        column_score = 0
        financial_columns = 0

        for col in df.columns:
            col_lower = str(col).lower()

            # Amount/money columns (high value)
            if any(keyword in col_lower for keyword in [
                'amount', 'cost', 'price', 'value', 'total', 'sum',
                'expense', 'revenue', 'sales', 'income', 'profit',
                'payment', 'charge', 'fee', 'bill', 'spend', 'money',
                'mrr', 'arr', 'cac', 'ltv', 'arpu'
            ]):
                financial_columns += 1
                column_score += 1.0

            # Date columns (medium value)
            elif any(keyword in col_lower for keyword in [
                'date', 'time', 'timestamp', 'created', 'updated',
                'transaction', 'purchase', 'order', 'invoice'
            ]):
                financial_columns += 1
                column_score += 0.7

            # Category columns (medium value)
            elif any(keyword in col_lower for keyword in [
                'category', 'type', 'class', 'group', 'department',
                'tag', 'label', 'classification'
            ]):
                financial_columns += 1
                column_score += 0.6

            # Description columns (low value)
            elif any(keyword in col_lower for keyword in [
                'description', 'desc', 'note', 'comment', 'memo',
                'detail', 'remark', 'info', 'text'
            ]):
                financial_columns += 1
                column_score += 0.3

        score += min(column_score, 3.0)

        # 3. Data content scoring (3 points max)
        content_score = 0

        # Check for numeric data (likely amounts)
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) > 0:
            content_score += 1.0

            # Check if numeric data looks like financial amounts
            for col in numeric_columns:
                values = df[col].dropna()
                if len(values) > 0:
                    # Check for typical financial ranges
                    if values.min() >= 0 and values.max() > 10:  # Positive amounts > $10
                        content_score += 0.5

                    # Check for decimal places (common in currency)
                    if any(val % 1 != 0 for val in values.head(10) if pd.notna(val)):
                        content_score += 0.3

        # Check for date-like data
        for col in df.columns:
            try:
                # Try to parse as dates
                date_series = pd.to_datetime(df[col], errors='coerce')
                if date_series.notna().sum() > len(df) * 0.5:  # >50% valid dates
                    content_score += 0.7
                    break
            except:
                continue

        score += min(content_score, 3.0)

        # 4. Data quality scoring (2 points max)
        quality_score = 0

        # Prefer sheets with more data
        if len(df) >= 10:
            quality_score += 0.5
        if len(df) >= 50:
            quality_score += 0.5
        if len(df) >= 100:
            quality_score += 0.5

        # Prefer sheets with fewer missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio < 0.1:  # <10% missing
            quality_score += 0.5
        elif missing_ratio < 0.3:  # <30% missing
            quality_score += 0.3

        score += min(quality_score, 2.0)

        # 5. Penalty for obviously non-financial sheets
        if any(keyword in sheet_name_lower for keyword in [
            'chart', 'graph', 'pivot', 'summary_stats', 'metadata',
            'config', 'settings', 'template', 'example'
        ]):
            score -= 1.0

        # Ensure score is within bounds
        return max(0.0, min(score, 10.0))

    def _generate_sheet_metadata(self, df: pd.DataFrame, sheet_name: str) -> dict:
        """Generate comprehensive metadata for a sheet to help LLM understand its content"""

        metadata = {
            'sheet_name': sheet_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns),
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'financial_score': self._score_sheet_for_financial_data(df, sheet_name),
            'content_summary': {},
            'relationships': {},
            'sample_data': {},
            'data_mess_analysis': self._analyze_data_mess(df)  # 😂 Tell LLM about the mess!
        }

        # Analyze each column
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            col_analysis = {
                'type': str(df[col].dtype),
                'non_null_count': len(col_data),
                'null_percentage': (len(df) - len(col_data)) / len(df) * 100,
                'unique_count': col_data.nunique(),
                'is_unique': col_data.nunique() == len(col_data)
            }

            # Add type-specific analysis
            if pd.api.types.is_numeric_dtype(df[col]):
                col_analysis.update({
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'has_decimals': any(val % 1 != 0 for val in col_data.head(20) if pd.notna(val))
                })

                # Check if it looks like financial data
                if col_analysis['min'] >= 0 and col_analysis['max'] > 10:
                    col_analysis['likely_financial'] = True

            elif pd.api.types.is_datetime64_any_dtype(df[col]) or self._looks_like_date(col_data):
                try:
                    date_series = pd.to_datetime(col_data, errors='coerce')
                    valid_dates = date_series.dropna()
                    if len(valid_dates) > 0:
                        col_analysis.update({
                            'date_range': {
                                'start': str(valid_dates.min()),
                                'end': str(valid_dates.max())
                            },
                            'is_date_column': True
                        })
                except:
                    pass
            else:
                # String/categorical analysis
                col_analysis.update({
                    'sample_values': list(col_data.head(5).astype(str)),
                    'most_common': col_data.value_counts().head(3).to_dict()
                })

            metadata['content_summary'][col] = col_analysis

        # Identify potential key columns (for joins)
        metadata['potential_keys'] = self._identify_potential_keys(df)

        # Generate natural language description
        metadata['description'] = self._generate_sheet_description(df, sheet_name, metadata)

        # Add sample rows for LLM context
        metadata['sample_data'] = df.head(3).to_dict('records')

        return metadata

    def _analyze_data_mess(self, df: pd.DataFrame) -> dict:
        """
        😂 Analyze the data mess and tell the LLM exactly what chaos it's dealing with!
        This helps the LLM understand data formatting issues and handle them properly.
        """
        mess_analysis = {
            'financial_formatting_issues': [],
            'data_type_problems': [],
            'sample_values': {},
            'cleaning_required': []
        }

        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Get sample values to show the LLM what we're dealing with
            sample_values = col_data.head(3).tolist()
            mess_analysis['sample_values'][col] = sample_values

            # Check for financial formatting mess
            if col_data.dtype == 'object':  # String columns
                has_dollar_signs = any('$' in str(val) for val in sample_values if val is not None)
                has_commas = any(',' in str(val) for val in sample_values if val is not None)
                has_mixed_formats = len(set(type(val).__name__ for val in sample_values)) > 1

                if has_dollar_signs or has_commas:
                    mess_analysis['financial_formatting_issues'].append({
                        'column': col,
                        'issues': {
                            'has_dollar_signs': has_dollar_signs,
                            'has_commas': has_commas,
                            'sample_messy_values': sample_values
                        }
                    })
                    mess_analysis['cleaning_required'].append(f"{col}: Remove $ and commas, convert to numeric")

                if has_mixed_formats:
                    mess_analysis['data_type_problems'].append({
                        'column': col,
                        'issue': 'Mixed data types',
                        'sample_types': [type(val).__name__ for val in sample_values]
                    })

            # Check for numeric columns that should be financial
            elif col_data.dtype in ['int64', 'float64']:
                # Check if values look like financial amounts
                if any(keyword in col.lower() for keyword in ['amount', 'cost', 'price', 'revenue', 'income', 'profit']):
                    if col_data.max() > 1000:  # Likely financial amounts
                        mess_analysis['cleaning_required'].append(f"{col}: Already numeric, format as currency for display")

        return mess_analysis

    def _select_primary_sheet(self, all_sheets_data: dict, sheet_metadata: dict) -> str:
        """Select primary sheet for backward compatibility"""

        # Score all sheets and return the highest scoring one
        best_sheet = None
        best_score = 0

        for sheet_name, metadata in sheet_metadata.items():
            score = metadata['financial_score']
            if score > best_score:
                best_score = score
                best_sheet = sheet_name

        return best_sheet or list(all_sheets_data.keys())[0]

    def _identify_potential_keys(self, df: pd.DataFrame) -> list:
        """Identify columns that could be used for joins"""

        potential_keys = []

        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            # Check if column could be a key
            col_lower = col.lower()

            # ID columns
            if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'number', 'ref']):
                potential_keys.append({
                    'column': col,
                    'type': 'id',
                    'unique_ratio': col_data.nunique() / len(col_data),
                    'sample_values': list(col_data.head(3).astype(str))
                })

            # Date columns (for time-based joins)
            elif any(keyword in col_lower for keyword in ['date', 'time', 'timestamp']):
                potential_keys.append({
                    'column': col,
                    'type': 'date',
                    'unique_ratio': col_data.nunique() / len(col_data),
                    'sample_values': list(col_data.head(3).astype(str))
                })

            # Category columns (for categorical joins)
            elif col_data.nunique() < len(col_data) * 0.5:  # Less than 50% unique
                potential_keys.append({
                    'column': col,
                    'type': 'category',
                    'unique_ratio': col_data.nunique() / len(col_data),
                    'sample_values': list(col_data.head(3).astype(str))
                })

        return potential_keys

    def _generate_sheet_description(self, df: pd.DataFrame, sheet_name: str, metadata: dict) -> str:
        """Generate natural language description of the sheet"""

        description = f"Sheet '{sheet_name}' contains {len(df)} rows and {len(df.columns)} columns. "

        # Identify the type of data
        financial_cols = []
        date_cols = []
        id_cols = []

        for col, analysis in metadata['content_summary'].items():
            if analysis.get('likely_financial'):
                financial_cols.append(col)
            elif analysis.get('is_date_column'):
                date_cols.append(col)
            elif analysis.get('is_unique') and 'id' in col.lower():
                id_cols.append(col)

        if financial_cols:
            description += f"Financial data columns: {', '.join(financial_cols)}. "
        if date_cols:
            description += f"Date columns: {', '.join(date_cols)}. "
        if id_cols:
            description += f"Identifier columns: {', '.join(id_cols)}. "

        # Suggest what this sheet might contain
        sheet_lower = sheet_name.lower()
        if any(keyword in sheet_lower for keyword in ['expense', 'cost', 'spend']):
            description += "This appears to be expense/cost data. "
        elif any(keyword in sheet_lower for keyword in ['revenue', 'sales', 'income']):
            description += "This appears to be revenue/sales data. "
        elif any(keyword in sheet_lower for keyword in ['customer', 'client']):
            description += "This appears to be customer data. "
        elif any(keyword in sheet_lower for keyword in ['product', 'item']):
            description += "This appears to be product data. "

        return description

    def _looks_like_date(self, series: pd.Series) -> bool:
        """Check if a series looks like it contains dates"""
        try:
            # Try to parse a few values as dates
            sample = series.head(10).astype(str)
            date_count = 0

            for val in sample:
                try:
                    pd.to_datetime(val)
                    date_count += 1
                except:
                    continue

            return date_count / len(sample) > 0.5
        except:
            return False

    def _analyze_sheet_relationships(self) -> dict:
        """Analyze potential relationships between sheets for joins"""

        if not hasattr(self, '_all_sheets_data') or not self._all_sheets_data:
            return {}

        relationships = {}
        sheet_names = list(self._all_sheets_data.keys())

        # Compare each pair of sheets
        for i, sheet1 in enumerate(sheet_names):
            for j, sheet2 in enumerate(sheet_names[i+1:], i+1):

                relationship = self._find_relationship_between_sheets(
                    sheet1, self._all_sheets_data[sheet1], self._sheet_metadata[sheet1],
                    sheet2, self._all_sheets_data[sheet2], self._sheet_metadata[sheet2]
                )

                if relationship:
                    relationships[f"{sheet1}__{sheet2}"] = relationship

        return relationships

    def _find_relationship_between_sheets(self, name1: str, df1: pd.DataFrame, meta1: dict,
                                        name2: str, df2: pd.DataFrame, meta2: dict) -> dict:
        """Find potential join relationships between two sheets"""

        potential_joins = []

        # Check for common columns (exact match)
        common_columns = set(df1.columns) & set(df2.columns)
        for col in common_columns:
            # Analyze if this column could be used for joining
            join_quality = self._assess_join_quality(df1[col], df2[col], col)
            if join_quality['viable']:
                potential_joins.append({
                    'type': 'exact_column_match',
                    'column1': col,
                    'column2': col,
                    'quality_score': join_quality['score'],
                    'match_ratio': join_quality['match_ratio'],
                    'description': f"Both sheets have '{col}' column with {join_quality['match_ratio']:.1%} matching values"
                })

        # Check for similar columns (fuzzy match)
        for col1 in df1.columns:
            for col2 in df2.columns:
                if col1 == col2:  # Already checked above
                    continue

                similarity = self._column_similarity(col1, col2, df1[col1], df2[col2])
                if similarity['score'] > 0.7:  # High similarity threshold
                    potential_joins.append({
                        'type': 'similar_columns',
                        'column1': col1,
                        'column2': col2,
                        'quality_score': similarity['score'],
                        'similarity_type': similarity['type'],
                        'description': f"'{col1}' and '{col2}' appear to contain similar data ({similarity['type']})"
                    })

        # Check for date-based relationships
        date_cols1 = [col for col, analysis in meta1['content_summary'].items()
                     if analysis.get('is_date_column')]
        date_cols2 = [col for col, analysis in meta2['content_summary'].items()
                     if analysis.get('is_date_column')]

        if date_cols1 and date_cols2:
            potential_joins.append({
                'type': 'temporal_relationship',
                'column1': date_cols1[0],
                'column2': date_cols2[0],
                'quality_score': 0.8,
                'description': f"Time-based relationship possible using date columns"
            })

        if not potential_joins:
            return None

        # Return the best relationship
        best_join = max(potential_joins, key=lambda x: x['quality_score'])

        return {
            'sheet1': name1,
            'sheet2': name2,
            'best_join': best_join,
            'all_possible_joins': potential_joins,
            'relationship_strength': best_join['quality_score']
        }

    def _assess_join_quality(self, series1: pd.Series, series2: pd.Series, column_name: str) -> dict:
        """Assess how good a column would be for joining"""

        # Clean the data
        clean1 = series1.dropna().astype(str)
        clean2 = series2.dropna().astype(str)

        if len(clean1) == 0 or len(clean2) == 0:
            return {'viable': False, 'score': 0, 'match_ratio': 0}

        # Find intersection
        set1 = set(clean1)
        set2 = set(clean2)
        intersection = set1 & set2

        if not intersection:
            return {'viable': False, 'score': 0, 'match_ratio': 0}

        # Calculate match ratio
        match_ratio = len(intersection) / max(len(set1), len(set2))

        # Score based on various factors
        score = 0

        # High match ratio is good
        score += match_ratio * 0.4

        # Unique values are better for joins
        uniqueness1 = len(set1) / len(clean1)
        uniqueness2 = len(set2) / len(clean2)
        avg_uniqueness = (uniqueness1 + uniqueness2) / 2
        score += avg_uniqueness * 0.3

        # ID-like columns are better
        col_lower = column_name.lower()
        if any(keyword in col_lower for keyword in ['id', 'key', 'code', 'number']):
            score += 0.3

        return {
            'viable': match_ratio > 0.1,  # At least 10% overlap
            'score': min(score, 1.0),
            'match_ratio': match_ratio
        }

    def _column_similarity(self, col1: str, col2: str, series1: pd.Series, series2: pd.Series) -> dict:
        """Check similarity between two columns"""

        # Name similarity
        name_sim = self._string_similarity(col1.lower(), col2.lower())

        # Data type similarity
        type_sim = 1.0 if str(series1.dtype) == str(series2.dtype) else 0.5

        # Value similarity (for categorical data)
        value_sim = 0
        try:
            if series1.dtype == 'object' and series2.dtype == 'object':
                clean1 = set(series1.dropna().astype(str))
                clean2 = set(series2.dropna().astype(str))
                if clean1 and clean2:
                    intersection = clean1 & clean2
                    value_sim = len(intersection) / max(len(clean1), len(clean2))
        except:
            pass

        # Combined score
        score = (name_sim * 0.4 + type_sim * 0.3 + value_sim * 0.3)

        # Determine similarity type
        if name_sim > 0.8:
            sim_type = "name_similarity"
        elif value_sim > 0.5:
            sim_type = "value_similarity"
        elif type_sim == 1.0:
            sim_type = "type_similarity"
        else:
            sim_type = "general_similarity"

        return {
            'score': score,
            'type': sim_type,
            'name_similarity': name_sim,
            'type_similarity': type_sim,
            'value_similarity': value_sim
        }

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using simple character overlap"""
        if not s1 or not s2:
            return 0.0

        # Simple character-based similarity
        set1 = set(s1)
        set2 = set(s2)
        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union) if union else 0.0
