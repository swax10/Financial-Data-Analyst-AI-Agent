"""
Smart Column Mapper Service
Automatically detects and maps column names to financial data types
"""

import re
import pandas as pd
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ColumnMapper:
    """
    Intelligently maps column names to financial data types regardless of naming conventions
    """

    def __init__(self):
        # Define patterns for different financial data types
        self.column_patterns = {
            'amount': {
                'patterns': [
                    r'amount', r'cost', r'price', r'value', r'total', r'sum',
                    r'expense', r'revenue', r'sales', r'income', r'profit',
                    r'payment', r'charge', r'fee', r'bill', r'spend',
                    r'money', r'cash', r'dollar', r'usd', r'eur', r'gbp',
                    r'debit', r'credit', r'balance', r'net', r'gross',
                    # Startup-specific patterns
                    r'mrr', r'arr', r'burn', r'runway', r'funding', r'valuation',
                    r'cac', r'ltv', r'arpu', r'acv', r'tcv', r'gmv',
                    r'churn.*rate', r'retention.*rate', r'conversion.*rate'
                ],
                'data_types': ['float64', 'int64', 'float32', 'int32'],
                'sample_checks': ['is_numeric', 'has_currency_symbols', 'is_startup_metric']
            },
            'date': {
                'patterns': [
                    r'date', r'time', r'timestamp', r'created', r'updated',
                    r'transaction.*date', r'purchase.*date', r'order.*date',
                    r'invoice.*date', r'payment.*date', r'due.*date',
                    r'start.*date', r'end.*date', r'period', r'when'
                ],
                'data_types': ['datetime64', 'object'],
                'sample_checks': ['is_date_like', 'has_date_patterns']
            },
            'category': {
                'patterns': [
                    r'category', r'type', r'class', r'group', r'kind',
                    r'department', r'division', r'section', r'tag',
                    r'label', r'classification', r'genre', r'style',
                    r'expense.*type', r'transaction.*type', r'payment.*type'
                ],
                'data_types': ['object', 'category'],
                'sample_checks': ['is_categorical', 'has_repeated_values']
            },
            'description': {
                'patterns': [
                    r'description', r'desc', r'note', r'comment', r'memo',
                    r'detail', r'remark', r'info', r'text', r'message',
                    r'narrative', r'explanation', r'summary', r'title'
                ],
                'data_types': ['object'],
                'sample_checks': ['is_text_like', 'has_varied_length']
            },
            'id': {
                'patterns': [
                    r'id', r'identifier', r'key', r'ref', r'reference',
                    r'number', r'num', r'code', r'index', r'serial',
                    r'transaction.*id', r'order.*id', r'invoice.*id'
                ],
                'data_types': ['int64', 'object', 'int32'],
                'sample_checks': ['is_unique_like', 'is_sequential']
            }
        }

    def analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze DataFrame columns and map them to financial data types
        """
        logger.info(f"Analyzing {len(df.columns)} columns for financial patterns")

        column_mapping = {}
        confidence_scores = {}

        for column in df.columns:
            mapping_result = self._analyze_single_column(df, column)
            column_mapping[column] = mapping_result['type']
            confidence_scores[column] = mapping_result['confidence']

        # Create reverse mapping for easy lookup
        type_to_columns = {}
        for col, col_type in column_mapping.items():
            if col_type not in type_to_columns:
                type_to_columns[col_type] = []
            type_to_columns[col_type].append({
                'name': col,
                'confidence': confidence_scores[col]
            })

        # Sort by confidence for each type
        for col_type in type_to_columns:
            type_to_columns[col_type].sort(key=lambda x: x['confidence'], reverse=True)

        result = {
            'column_mapping': column_mapping,
            'confidence_scores': confidence_scores,
            'type_to_columns': type_to_columns,
            'best_columns': self._get_best_columns(type_to_columns)
        }

        logger.info(f"Column analysis complete. Best columns: {result['best_columns']}")
        return result

    def _analyze_single_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Analyze a single column to determine its financial data type
        """
        column_lower = column.lower().strip()
        data_type = str(df[column].dtype)
        sample_data = df[column].dropna().head(10)

        scores = {}

        # Check each financial data type
        for fin_type, config in self.column_patterns.items():
            score = 0

            # 1. Pattern matching (40% weight)
            pattern_score = self._check_name_patterns(column_lower, config['patterns'])
            score += pattern_score * 0.4

            # 2. Data type matching (20% weight)
            dtype_score = 1.0 if data_type in config['data_types'] else 0.0
            score += dtype_score * 0.2

            # 3. Sample data analysis (40% weight)
            sample_score = self._check_sample_data(sample_data, config['sample_checks'], fin_type)
            score += sample_score * 0.4

            scores[fin_type] = score

        # Find best match
        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        # If confidence is too low, mark as 'unknown'
        if confidence < 0.3:
            best_type = 'unknown'
            confidence = 0.0

        return {
            'type': best_type,
            'confidence': confidence,
            'all_scores': scores
        }

    def _check_name_patterns(self, column_name: str, patterns: List[str]) -> float:
        """
        Check if column name matches any of the patterns
        """
        for pattern in patterns:
            if re.search(pattern, column_name, re.IGNORECASE):
                return 1.0
        return 0.0

    def _check_sample_data(self, sample_data: pd.Series, checks: List[str], fin_type: str) -> float:
        """
        Analyze sample data to determine if it matches the financial type
        """
        if len(sample_data) == 0:
            return 0.0

        score = 0.0
        check_count = len(checks)

        for check in checks:
            if check == 'is_numeric':
                score += self._is_numeric_data(sample_data)
            elif check == 'has_currency_symbols':
                score += self._has_currency_symbols(sample_data)
            elif check == 'is_date_like':
                score += self._is_date_like(sample_data)
            elif check == 'has_date_patterns':
                score += self._has_date_patterns(sample_data)
            elif check == 'is_categorical':
                score += self._is_categorical_data(sample_data)
            elif check == 'has_repeated_values':
                score += self._has_repeated_values(sample_data)
            elif check == 'is_text_like':
                score += self._is_text_like(sample_data)
            elif check == 'has_varied_length':
                score += self._has_varied_length(sample_data)
            elif check == 'is_unique_like':
                score += self._is_unique_like(sample_data)
            elif check == 'is_sequential':
                score += self._is_sequential(sample_data)
            elif check == 'is_startup_metric':
                score += self._is_startup_metric(sample_data)

        return score / check_count if check_count > 0 else 0.0

    def _is_numeric_data(self, data: pd.Series) -> float:
        """Check if data is numeric"""
        try:
            pd.to_numeric(data, errors='coerce')
            numeric_count = pd.to_numeric(data, errors='coerce').notna().sum()
            return numeric_count / len(data)
        except:
            return 0.0

    def _has_currency_symbols(self, data: pd.Series) -> float:
        """Check if data contains currency symbols"""
        currency_pattern = r'[\$€£¥₹₽]'
        matches = data.astype(str).str.contains(currency_pattern, na=False).sum()
        return min(matches / len(data), 1.0)

    def _is_date_like(self, data: pd.Series) -> float:
        """Check if data looks like dates"""
        try:
            pd.to_datetime(data, errors='coerce')
            date_count = pd.to_datetime(data, errors='coerce').notna().sum()
            return date_count / len(data)
        except:
            return 0.0

    def _has_date_patterns(self, data: pd.Series) -> float:
        """Check if data has date-like patterns"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]

        matches = 0
        for pattern in date_patterns:
            matches += data.astype(str).str.contains(pattern, na=False).sum()

        return min(matches / len(data), 1.0)

    def _is_categorical_data(self, data: pd.Series) -> float:
        """Check if data is categorical"""
        unique_ratio = data.nunique() / len(data)
        # Good categorical data has low unique ratio
        return 1.0 - min(unique_ratio, 1.0)

    def _has_repeated_values(self, data: pd.Series) -> float:
        """Check if data has repeated values (good for categories)"""
        value_counts = data.value_counts()
        repeated_count = (value_counts > 1).sum()
        return min(repeated_count / data.nunique(), 1.0)

    def _is_text_like(self, data: pd.Series) -> float:
        """Check if data is text-like"""
        if data.dtype == 'object':
            # Check if values are strings with reasonable length
            text_count = data.astype(str).str.len().between(5, 200).sum()
            return text_count / len(data)
        return 0.0

    def _has_varied_length(self, data: pd.Series) -> float:
        """Check if text data has varied length (good for descriptions)"""
        if data.dtype == 'object':
            lengths = data.astype(str).str.len()
            length_std = lengths.std()
            return min(length_std / 50, 1.0)  # Normalize by expected variation
        return 0.0

    def _is_unique_like(self, data: pd.Series) -> float:
        """Check if data is mostly unique (good for IDs)"""
        unique_ratio = data.nunique() / len(data)
        return unique_ratio

    def _is_sequential(self, data: pd.Series) -> float:
        """Check if data is sequential (good for IDs)"""
        try:
            numeric_data = pd.to_numeric(data, errors='coerce').dropna()
            if len(numeric_data) < 2:
                return 0.0

            diffs = numeric_data.diff().dropna()
            # Check if differences are mostly 1 (sequential)
            sequential_count = (diffs == 1).sum()
            return sequential_count / len(diffs)
        except:
            return 0.0

    def _get_best_columns(self, type_to_columns: Dict[str, List[Dict]]) -> Dict[str, Optional[str]]:
        """
        Get the best column for each financial data type
        """
        best_columns = {}

        for fin_type in ['amount', 'date', 'category', 'description', 'id']:
            if fin_type in type_to_columns and type_to_columns[fin_type]:
                best_col = type_to_columns[fin_type][0]  # Highest confidence
                if best_col['confidence'] > 0.5:  # Only if confident enough
                    best_columns[fin_type] = best_col['name']
                else:
                    best_columns[fin_type] = None
            else:
                best_columns[fin_type] = None

        return best_columns

    def get_column_for_type(self, analysis_result: Dict[str, Any],
                           desired_type: str) -> Optional[str]:
        """
        Get the best column name for a desired financial data type
        """
        best_columns = analysis_result.get('best_columns', {})
        return best_columns.get(desired_type)

    def generate_dynamic_code(self, analysis_result: Dict[str, Any],
                            query_type: str, file_path: str) -> str:
        """
        Generate Python code using the detected column names
        """
        best_cols = analysis_result['best_columns']

        # Get the actual column names
        amount_col = best_cols.get('amount', 'Amount')  # Fallback to 'Amount'
        date_col = best_cols.get('date', 'Date')
        category_col = best_cols.get('category', 'Category')

        if query_type == 'total_spending':
            return f"""
import pandas as pd

# Load the data
df = pd.read_csv('{file_path}')

# Use detected column: {amount_col}
total = df['{amount_col}'].sum()
average = df['{amount_col}'].mean()
count = len(df)

print(f"Total: ${{total:,.2f}}")
print(f"Average: ${{average:,.2f}}")
print(f"Transactions: {{count}}")
"""

        elif query_type == 'category_analysis' and category_col:
            return f"""
import pandas as pd

# Load the data
df = pd.read_csv('{file_path}')

# Use detected columns: {amount_col}, {category_col}
category_summary = df.groupby('{category_col}')['{amount_col}'].agg(['sum', 'mean', 'count'])
print("Category Analysis:")
print(category_summary.sort_values('sum', ascending=False))
"""

        # Add more query types as needed
        return self._generate_fallback_code(file_path, best_cols)

    def _is_startup_metric(self, data: pd.Series) -> float:
        """Check if data contains startup-specific metrics"""
        try:
            # Convert to numeric if possible
            numeric_data = pd.to_numeric(data, errors='coerce')

            # Check for typical startup metric ranges
            if numeric_data.isna().all():
                return 0.0

            # Remove NaN values
            clean_data = numeric_data.dropna()
            if len(clean_data) == 0:
                return 0.0

            # Startup metrics often have specific characteristics
            score = 0.0

            # 1. MRR/ARR typically in thousands to millions
            if clean_data.min() >= 1000 and clean_data.max() <= 10000000:
                score += 0.3

            # 2. Rates (churn, conversion) typically 0-100 or 0-1
            if clean_data.min() >= 0 and clean_data.max() <= 100:
                score += 0.2

            # 3. CAC/LTV typically in hundreds to thousands
            if clean_data.min() >= 10 and clean_data.max() <= 50000:
                score += 0.2

            # 4. Growth rates often show percentage patterns
            if any(val > 1 and val < 1000 for val in clean_data):
                score += 0.3

            return min(score, 1.0)

        except Exception:
            return 0.0
