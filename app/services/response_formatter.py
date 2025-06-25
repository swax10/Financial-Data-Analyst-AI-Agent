"""
Response Formatter Service
Converts technical analysis output into human-friendly responses
"""

import re
import json
from typing import Dict, List, Any, Optional


class ResponseFormatter:
    """
    Converts technical analysis output into conversational, human-friendly responses
    Uses dual LLM approach: LLM #1 for analysis, LLM #2 for executive communication
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.emojis = {
            'money': '💰', 'chart': '📊', 'up': '📈', 'down': '📉',
            'food': '🍔', 'transport': '🚗', 'entertainment': '🎬',
            'shopping': '🛍️', 'bills': '💡', 'health': '🏥',
            'celebration': '🎉', 'warning': '⚠️', 'info': 'ℹ️'
        }

    async def format_analysis_response(self, query: str, execution_result: Dict[str, Any],
                                      generated_code: str) -> str:
        """
        Convert technical execution results into human-friendly responses using dual LLM approach
        """

        print("🔄" * 60)
        print("🔄 RESPONSE FORMATTER: Starting format_analysis_response")
        print("🔄" * 60)
        print(f"📝 Query: {query}")
        print(f"📊 Execution result keys: {list(execution_result.keys())}")
        print(f"💻 Generated code length: {len(generated_code)}")
        print(f"🎯 LLM service available: {self.llm_service is not None}")

        # Extract technical results
        technical_results = self._extract_technical_results(execution_result)
        execution_success = execution_result.get('success', False)

        print(f"📊 Technical results length: {len(technical_results)}")
        print(f"✅ Execution success: {execution_success}")
        print(f"📋 Technical results preview: {technical_results[:200]}...")

        # Use dual LLM approach if LLM service is available
        if self.llm_service:
            print("🎩" * 40)
            print("🎩 DUAL LLM APPROACH: LLM service is available!")
            print("🎩" * 40)

            try:
                print("🎩 CALLING LLM #2 FOR EXECUTIVE SUMMARY...")
                print(f"📝 Query: {query}")
                print(f"📊 Technical results length: {len(technical_results)}")
                print(f"✅ Execution success: {execution_success}")

                # LLM #2: Convert technical results to executive summary
                print("🎩 Invoking generate_executive_summary method...")
                summary_result = await self.llm_service.generate_executive_summary(
                    user_query=query,
                    technical_results=technical_results,
                    execution_success=execution_success
                )

                print(f"🎩 LLM #2 RESULT RECEIVED!")
                print(f"🎩 Result keys: {list(summary_result.keys())}")
                print(f"🎩 Success: {summary_result.get('success', False)}")

                if summary_result.get('success'):
                    executive_summary = summary_result['executive_summary']
                    print(f"🎩 EXECUTIVE SUMMARY LENGTH: {len(executive_summary)}")
                    print(f"🎩 EXECUTIVE SUMMARY PREVIEW: {executive_summary[:300]}...")
                    print("🎉 DUAL LLM SUCCESS! Returning executive summary.")
                    return executive_summary
                else:
                    print(f"❌ LLM #2 FAILED: {summary_result.get('error', 'Unknown error')}")
                    print("🔄 Falling back to enhanced formatting...")
                    # Fallback to enhanced formatting if LLM #2 fails
                    return self._format_with_enhanced_fallback(query, technical_results, execution_success)

            except Exception as e:
                print(f"❌ EXCEPTION IN LLM #2: {str(e)}")
                print("🔄 Falling back to enhanced formatting due to exception...")
                import traceback
                traceback.print_exc()
                # Fallback to enhanced formatting if LLM #2 fails
                return self._format_with_enhanced_fallback(query, technical_results, execution_success)
        else:
            print("⚠️" * 40)
            print("⚠️ NO LLM SERVICE AVAILABLE!")
            print("⚠️ Using legacy pattern-based formatting...")
            print("⚠️" * 40)

        # Fallback to legacy formatting if no LLM service
        print("🔄 Using legacy pattern-based formatting...")
        return self._format_with_legacy_patterns(query, technical_results)

    def _extract_technical_results(self, execution_result: Dict[str, Any]) -> str:
        """Extract and combine all technical output from code execution"""

        # Extract stdout output
        stdout_lines = execution_result.get('logs', {}).get('stdout', [])
        results = execution_result.get('results', [])

        # Combine all output
        all_output = '\n'.join(stdout_lines)
        if results:
            for result in results:
                if isinstance(result, dict) and 'text' in result:
                    all_output += '\n' + str(result['text'])
                elif isinstance(result, str):
                    all_output += '\n' + result
                else:
                    all_output += '\n' + str(result)

        return all_output.strip()

    def _format_with_enhanced_fallback(self, query: str, technical_results: str, execution_success: bool) -> str:
        """Enhanced fallback formatting when LLM #2 is not available"""

        if not execution_success:
            return f"""
❌ **Analysis Error**

I encountered an issue while analyzing your financial data for: "{query}"

**Error Details:**
{technical_results[:300]}{'...' if len(technical_results) > 300 else ''}

💡 **Suggestions:**
• Check if your data has the required columns
• Ensure date and amount columns are properly formatted
• Try a more specific query

Would you like me to help you explore your data structure first?
"""

        if not technical_results or len(technical_results.strip()) < 20:
            return f"""
📊 **Financial Analysis Results**

I've analyzed your data for: "{query}"

The analysis completed successfully, but the results were minimal. This might mean:
• The data doesn't contain relevant information for this query
• The query might need to be more specific
• The data might need different formatting

💡 **Try asking:**
• "What columns are in my data?"
• "Show me a summary of my financial data"
• "What time period does my data cover?"
"""

        # Extract meaningful insights from technical results
        insights = self._extract_meaningful_insights(technical_results)

        return f"""
💼 **Financial Analysis Results**

Based on your query: "{query}"

📊 **Key Findings:**
{insights}

📋 **Technical Details:**
{technical_results[:400]}{'...' if len(technical_results) > 400 else ''}

💡 **Next Steps:**
Ask me specific questions about the metrics you're most interested in, such as trends, comparisons, or deeper analysis.
"""

    def _extract_meaningful_insights(self, technical_results: str) -> str:
        """Extract meaningful business insights from technical output"""

        insights = []
        lines = technical_results.split('\n')

        for line in lines:
            line = line.strip()

            # Look for summary statistics
            if 'mean' in line.lower() and any(char.isdigit() for char in line):
                insights.append(f"• Average value identified in the data")
            elif 'sum' in line.lower() and any(char.isdigit() for char in line):
                insights.append(f"• Total amounts calculated")
            elif 'count' in line.lower() and any(char.isdigit() for char in line):
                insights.append(f"• Record counts analyzed")
            elif 'max' in line.lower() and any(char.isdigit() for char in line):
                insights.append(f"• Maximum values identified")
            elif 'min' in line.lower() and any(char.isdigit() for char in line):
                insights.append(f"• Minimum values found")

            # Look for data shape information
            elif 'shape:' in line.lower():
                insights.append(f"• Dataset dimensions: {line.split(':')[1].strip()}")
            elif 'columns:' in line.lower():
                insights.append(f"• Data structure analyzed")

        if not insights:
            insights.append("• Analysis completed with detailed results")
            insights.append("• Multiple data points processed")

        return '\n'.join(insights[:5])  # Limit to top 5 insights

    def _format_with_legacy_patterns(self, query: str, technical_results: str) -> str:
        """Legacy pattern-based formatting (original approach)"""

        # Determine query type and format accordingly
        query_lower = query.lower()

        if 'total' in query_lower and ('amount' in query_lower or 'spent' in query_lower):
            return self._format_total_spending(technical_results, query)
        elif 'category' in query_lower or 'categories' in query_lower:
            return self._format_category_analysis(technical_results, query)
        elif 'month' in query_lower or 'trend' in query_lower:
            return self._format_trend_analysis(technical_results, query, [])
        elif 'highest' in query_lower or 'most' in query_lower:
            return self._format_highest_analysis(technical_results, query)
        elif any(keyword in query_lower for keyword in ['mrr', 'arr', 'growth', 'revenue']):
            return self._format_growth_metrics(technical_results, query)
        elif any(keyword in query_lower for keyword in ['cac', 'ltv', 'unit economics', 'payback']):
            return self._format_unit_economics(technical_results, query)
        elif any(keyword in query_lower for keyword in ['burn', 'runway', 'cash', 'funding']):
            return self._format_burn_runway(technical_results, query)
        else:
            return self._format_general_analysis(technical_results, query)

    def _format_total_spending(self, output: str, query: str) -> str:
        """Format total spending responses"""

        # Extract total amount
        total_match = re.search(r'Total expenses?: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        avg_match = re.search(r'Average expense: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        count_match = re.search(r'Number of transactions: (\d+)', output, re.IGNORECASE)

        response = f"{self.emojis['money']} **Financial Summary**\n\n"

        if total_match:
            total = total_match.group(1)
            response += f"Your total spending is **${total}**! "

            if avg_match:
                avg = avg_match.group(1)
                response += f"That's an average of **${avg}** per transaction. "

            if count_match:
                count = count_match.group(1)
                response += f"You had **{count} transactions** in total."
        else:
            response += "I analyzed your spending data, but couldn't extract the total amount. Here's what I found:\n\n"
            response += self._extract_key_insights(output)

        return response

    def _format_category_analysis(self, output: str, query: str) -> str:
        """Format category analysis responses"""

        response = f"{self.emojis['chart']} **Spending by Category**\n\n"

        # Look for category breakdown
        if "Expenses by Category" in output:
            # Extract category data
            category_section = self._extract_section(output, "Expenses by Category")
            if category_section:
                response += "Here's how your spending breaks down:\n\n"

                # Parse category data (simplified)
                lines = category_section.split('\n')
                for line in lines[1:6]:  # Take first 5 categories
                    if line.strip() and not line.startswith('='):
                        # Try to extract category and amount
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            category = parts[0]
                            emoji = self._get_category_emoji(category)
                            response += f"{emoji} **{category.title()}**: Found in your data\n"

            # Add total if available
            total_match = re.search(r'Total expenses?: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
            if total_match:
                total = total_match.group(1)
                response += f"\n{self.emojis['money']} **Total across all categories: ${total}**"
        else:
            response += "I analyzed your spending categories. Here are the key insights:\n\n"
            response += self._extract_key_insights(output)

        return response

    def _format_trend_analysis(self, output: str, query: str, results: List[Any]) -> str:
        """Format trend analysis responses"""

        response = f"{self.emojis['chart']} **Spending Trends Analysis**\n\n"

        # Check for trend data in results
        if results:
            for result in results:
                if isinstance(result, dict) and 'text' in result:
                    result_text = result['text']
                    if 'sum' in result_text and 'Period' in result_text:
                        # Parse the trend data
                        try:
                            # Extract period data
                            periods = self._extract_period_data(result_text)
                            if periods:
                                response += "Here's your spending over time:\n\n"

                                max_period = max(periods.items(), key=lambda x: x[1])
                                min_period = min(periods.items(), key=lambda x: x[1])

                                response += f"{self.emojis['up']} **Highest spending**: {max_period[0]} with **${max_period[1]:.2f}**\n"
                                response += f"{self.emojis['down']} **Lowest spending**: {min_period[0]} with **${min_period[1]:.2f}**\n\n"

                                total_trend = sum(periods.values())
                                response += f"{self.emojis['money']} **Total across all periods: ${total_trend:.2f}**"

                                return response
                        except:
                            pass

        # Fallback to general trend analysis
        if "Trend Analysis" in output:
            response += "I've analyzed your spending trends over time. "

            # Look for specific insights
            if "monthly" in query.lower():
                response += "Here's your monthly breakdown:\n\n"
            elif "quarterly" in query.lower():
                response += "Here's your quarterly breakdown:\n\n"

            response += self._extract_key_insights(output)
        else:
            response += "I analyzed your spending patterns over time:\n\n"
            response += self._extract_key_insights(output)

        return response

    def _format_highest_analysis(self, output: str, query: str) -> str:
        """Format highest/most analysis responses"""

        if 'month' in query.lower():
            response = f"{self.emojis['up']} **Highest Spending Month**\n\n"
        elif 'category' in query.lower():
            response = f"{self.emojis['chart']} **Top Spending Categories**\n\n"
        else:
            response = f"{self.emojis['info']} **Top Spending Analysis**\n\n"

        # Extract insights
        response += self._extract_key_insights(output)

        return response

    def _format_general_analysis(self, output: str, query: str) -> str:
        """Format general analysis responses"""

        response = f"{self.emojis['chart']} **Financial Analysis Results**\n\n"
        response += "Here's what I found in your data:\n\n"
        response += self._extract_key_insights(output)

        return response

    def _extract_key_insights(self, output: str) -> str:
        """Extract key insights from technical output"""

        insights = []

        # If output is too short or generic, return the actual output
        if len(output.strip()) < 50:
            return "✅ Analysis completed successfully!\n📋 Check the detailed breakdown above for more information."

        # Extract total amounts (multiple patterns)
        total_patterns = [
            r'Total expenses?: \$?([\d,]+\.?\d*)',
            r'Total.*?: \$?([\d,]+\.?\d*)',
            r'Sum.*?: \$?([\d,]+\.?\d*)',
            r'Total: ([\d,]+\.?\d*)',
            r'(\d+\.?\d*)\s*total'
        ]

        for pattern in total_patterns:
            total_match = re.search(pattern, output, re.IGNORECASE)
            if total_match:
                insights.append(f"{self.emojis['money']} Total amount: **${total_match.group(1)}**")
                break

        # Extract averages (multiple patterns)
        avg_patterns = [
            r'Average.*?: \$?([\d,]+\.?\d*)',
            r'Mean.*?: \$?([\d,]+\.?\d*)',
            r'Avg.*?: \$?([\d,]+\.?\d*)'
        ]

        for pattern in avg_patterns:
            avg_match = re.search(pattern, output, re.IGNORECASE)
            if avg_match:
                insights.append(f"📊 Average: **${avg_match.group(1)}**")
                break

        # Extract counts (multiple patterns)
        count_patterns = [
            r'Number of.*?: (\d+)',
            r'Count.*?: (\d+)',
            r'Total.*?transactions.*?: (\d+)',
            r'(\d+)\s*rows?',
            r'(\d+)\s*records?'
        ]

        for pattern in count_patterns:
            count_match = re.search(pattern, output, re.IGNORECASE)
            if count_match:
                insights.append(f"🔢 Count: **{count_match.group(1)}**")
                break

        # Extract key statistics from pandas output
        if 'describe()' in output or 'Summary Statistics' in output:
            # Look for mean, std, min, max values
            stats_patterns = [
                (r'mean\s+([0-9.]+)', 'Average'),
                (r'std\s+([0-9.]+)', 'Standard Deviation'),
                (r'min\s+([0-9.]+)', 'Minimum'),
                (r'max\s+([0-9.]+)', 'Maximum'),
                (r'50%\s+([0-9.]+)', 'Median')
            ]

            for pattern, label in stats_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    insights.append(f"📈 {label}: **{match.group(1)}**")

        # If we found specific insights, return them
        if insights:
            return '\n'.join(insights)

        # Otherwise, try to extract meaningful lines from the output
        lines = output.strip().split('\n')
        meaningful_lines = []

        for line in lines:
            line = line.strip()
            # Skip empty lines, headers, and technical output
            if (line and
                not line.startswith('===') and
                not line.startswith('---') and
                not line.startswith('Shape:') and
                not line.startswith('Columns:') and
                not line.startswith('dtype') and
                len(line) > 10):
                meaningful_lines.append(line)

        # Return the most meaningful lines
        if meaningful_lines:
            # Take first few meaningful lines
            selected_lines = meaningful_lines[:5]
            return '\n'.join(f"📋 {line}" for line in selected_lines)

        # Final fallback - return actual output if it's substantial
        if len(output.strip()) > 100:
            # Clean up the output and return first part
            clean_output = output.strip()[:500] + "..." if len(output.strip()) > 500 else output.strip()
            return f"📊 **Analysis Results:**\n\n{clean_output}"

        # Last resort fallback
        return "✅ Analysis completed successfully!\n📋 Check the detailed breakdown above for more information."

    def _extract_section(self, output: str, section_name: str) -> Optional[str]:
        """Extract a specific section from output"""

        lines = output.split('\n')
        in_section = False
        section_lines = []

        for line in lines:
            if section_name in line:
                in_section = True
                continue
            elif in_section and line.startswith('==='):
                break
            elif in_section:
                section_lines.append(line)

        return '\n'.join(section_lines) if section_lines else None

    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for category"""

        category_lower = category.lower()

        if 'food' in category_lower or 'grocery' in category_lower or 'restaurant' in category_lower:
            return self.emojis['food']
        elif 'transport' in category_lower or 'gas' in category_lower or 'car' in category_lower:
            return self.emojis['transport']
        elif 'entertainment' in category_lower or 'movie' in category_lower:
            return self.emojis['entertainment']
        elif 'shopping' in category_lower or 'retail' in category_lower:
            return self.emojis['shopping']
        elif 'bill' in category_lower or 'utility' in category_lower:
            return self.emojis['bills']
        elif 'health' in category_lower or 'medical' in category_lower:
            return self.emojis['health']
        else:
            return self.emojis['chart']

    def _extract_period_data(self, result_text: str) -> Dict[str, float]:
        """Extract period data from result text"""

        periods = {}

        # Look for period patterns like "Period('2024-01', 'M'): 100.5"
        pattern = r"Period\('(\d{4}-\d{2})', 'M'\): ([\d.]+)"
        matches = re.findall(pattern, result_text)

        for period, amount in matches:
            # Convert to readable format
            year, month = period.split('-')
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_name = month_names[int(month) - 1]
            readable_period = f"{month_name} {year}"
            periods[readable_period] = float(amount)

        return periods

    def _format_growth_metrics(self, output: str, query: str) -> str:
        """Format growth metrics responses"""

        response = f"🚀 **Growth Metrics Analysis**\n\n"

        # Extract MRR
        mrr_match = re.search(r'Current MRR: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if mrr_match:
            mrr = mrr_match.group(1)
            response += f"💰 **Monthly Recurring Revenue (MRR)**: ${mrr}\n"

        # Extract ARR
        arr_match = re.search(r'Projected ARR: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if arr_match:
            arr = arr_match.group(1)
            response += f"📈 **Annual Recurring Revenue (ARR)**: ${arr}\n"

        # Extract growth rate
        growth_match = re.search(r'Month-over-Month Growth: ([\d.-]+)%', output, re.IGNORECASE)
        if growth_match:
            growth = float(growth_match.group(1))
            if growth > 0:
                response += f"🔥 **Month-over-Month Growth**: +{growth:.1f}% (Growing!)\n"
            else:
                response += f"📉 **Month-over-Month Growth**: {growth:.1f}% (Declining)\n"

        # Extract average growth
        avg_growth_match = re.search(r'Average Monthly Growth Rate: ([\d.-]+)%', output, re.IGNORECASE)
        if avg_growth_match:
            avg_growth = float(avg_growth_match.group(1))
            response += f"📊 **Average Growth Rate**: {avg_growth:.1f}% per month\n"

        response += f"\n{self._extract_key_insights(output)}"

        return response

    def _format_unit_economics(self, output: str, query: str) -> str:
        """Format unit economics responses"""

        response = f"💎 **Unit Economics Analysis**\n\n"

        # Extract CAC
        cac_match = re.search(r'Customer Acquisition Cost \(CAC\): \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if cac_match:
            cac = cac_match.group(1)
            response += f"💰 **Customer Acquisition Cost (CAC)**: ${cac}\n"

        # Extract LTV
        ltv_match = re.search(r'Customer Lifetime Value \(LTV\): \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if ltv_match:
            ltv = ltv_match.group(1)
            response += f"💎 **Customer Lifetime Value (LTV)**: ${ltv}\n"

        # Extract LTV:CAC ratio
        ratio_match = re.search(r'LTV:CAC Ratio: ([\d.]+)x', output, re.IGNORECASE)
        if ratio_match:
            ratio = float(ratio_match.group(1))
            if ratio >= 3:
                response += f"🎉 **LTV:CAC Ratio**: {ratio:.1f}x (Excellent!)\n"
            elif ratio >= 2:
                response += f"✅ **LTV:CAC Ratio**: {ratio:.1f}x (Good)\n"
            else:
                response += f"⚠️ **LTV:CAC Ratio**: {ratio:.1f}x (Needs improvement)\n"

        # Extract ARPU
        arpu_match = re.search(r'Average Revenue Per User \(ARPU\): \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if arpu_match:
            arpu = arpu_match.group(1)
            response += f"📈 **Average Revenue Per User (ARPU)**: ${arpu}\n"

        # Extract payback period
        payback_match = re.search(r'Payback Period: ([\d.]+) months', output, re.IGNORECASE)
        if payback_match:
            payback = float(payback_match.group(1))
            if payback <= 12:
                response += f"⚡ **Payback Period**: {payback:.1f} months (Great!)\n"
            elif payback <= 24:
                response += f"⏰ **Payback Period**: {payback:.1f} months (Acceptable)\n"
            else:
                response += f"🐌 **Payback Period**: {payback:.1f} months (Too long)\n"

        # Extract gross margin
        margin_match = re.search(r'Gross Margin: ([\d.]+)%', output, re.IGNORECASE)
        if margin_match:
            margin = float(margin_match.group(1))
            if margin >= 70:
                response += f"💹 **Gross Margin**: {margin:.1f}% (Excellent!)\n"
            elif margin >= 50:
                response += f"📊 **Gross Margin**: {margin:.1f}% (Good)\n"
            else:
                response += f"📉 **Gross Margin**: {margin:.1f}% (Needs improvement)\n"

        return response

    def _format_burn_runway(self, output: str, query: str) -> str:
        """Format burn rate and runway responses"""

        response = f"🔥 **Burn Rate & Runway Analysis**\n\n"

        # Extract burn rate
        burn_match = re.search(r'Current Monthly Burn Rate: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if burn_match:
            burn = burn_match.group(1)
            response += f"🔥 **Monthly Burn Rate**: ${burn}\n"

        # Extract cash balance
        cash_match = re.search(r'Current Cash Balance: \$?([\d,]+\.?\d*)', output, re.IGNORECASE)
        if cash_match:
            cash = cash_match.group(1)
            response += f"💰 **Cash Balance**: ${cash}\n"

        # Extract runway
        runway_match = re.search(r'Runway: ([\d.]+) months', output, re.IGNORECASE)
        if runway_match:
            runway = float(runway_match.group(1))

            if runway >= 18:
                response += f"🛣️ **Runway**: {runway:.1f} months (Healthy! 💪)\n"
            elif runway >= 12:
                response += f"🛣️ **Runway**: {runway:.1f} months (Moderate ⚠️)\n"
            elif runway >= 6:
                response += f"🛣️ **Runway**: {runway:.1f} months (Low - Consider fundraising! 🚨)\n"
            else:
                response += f"🛣️ **Runway**: {runway:.1f} months (Critical - Immediate action needed! 🆘)\n"

        # Extract runway date
        date_match = re.search(r'Estimated Cash Depletion: ([A-Za-z]+ \d{4})', output, re.IGNORECASE)
        if date_match:
            date = date_match.group(1)
            response += f"📅 **Cash Depletion Date**: {date}\n"

        # Extract burn trend
        if "increasing" in output.lower():
            response += f"📈 **Burn Trend**: Increasing (Monitor closely!)\n"
        elif "decreasing" in output.lower():
            response += f"📉 **Burn Trend**: Decreasing (Good progress!)\n"
        elif "stable" in output.lower():
            response += f"➡️ **Burn Trend**: Stable\n"

        # Check for profitability
        if "Profitable!" in output:
            response += f"🎉 **Status**: Profitable! No runway concerns.\n"

        return response
