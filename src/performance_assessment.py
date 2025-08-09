"""
Performance Assessment Module for Differential Privacy System
Measures: Latency, CPU Utilization, Privacy Level, and Utility
"""

import time
import psutil
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sqlalchemy import text
import json
from datetime import datetime

# Import the shared queries from dp_system
from dp_system import DifferentialPrivacySystem, DEFAULT_TEST_QUERIES
from src.aes_system import AESEncryptionSystem

GLOBAL_NUM_ROWS = 1000

class AESPerformanceAssessment:
    """Performance assessment for AES encryption system."""
    def __init__(self, csv_path="dataset/adult.csv", columns=None, num_rows=GLOBAL_NUM_ROWS):
        if columns is None:
            columns = ['education', 'occupation']
        self.csv_path = csv_path
        self.columns = columns
        self.num_rows = num_rows
        from dp_system import DEFAULT_TEST_QUERIES
        self.test_queries = DEFAULT_TEST_QUERIES.copy()

    def run_performance_test(self, num_iterations=3):
        import time, psutil, sqlite3
        results = []
        for i in range(num_iterations):
            df = pd.read_csv(self.csv_path).head(self.num_rows)
            aes = AESEncryptionSystem()

            # Measure encryption time
            start_enc = time.time()
            encrypted_df = aes.encrypt_dataframe(df, self.columns)
            encryption_time = time.time() - start_enc

            # Measure decryption time
            start_dec = time.time()
            decrypted_df = aes.decrypt_dataframe(encrypted_df, self.columns)
            decryption_time = time.time() - start_dec

            # Load both original and decrypted into in-memory SQLite for fair SQL querying
            conn_orig = sqlite3.connect(':memory:')
            conn_dec = sqlite3.connect(':memory:')
            df.to_sql('census_income', conn_orig, index=False, if_exists='replace')
            decrypted_df.to_sql('census_income', conn_dec, index=False, if_exists='replace')

            for query in self.test_queries:
                # Measure query execution time on decrypted data
                start_query = time.time()
                dec_result = pd.read_sql(query, conn_dec)
                query_time = time.time() - start_query

                # For utility: run on original data
                orig_result = pd.read_sql(query, conn_orig)

                # Utility: compare results (for COUNT/SUM/AVG, compare first value)
                try:
                    orig_val = orig_result.iloc[0, 0]
                    dec_val = dec_result.iloc[0, 0]
                    if orig_val != 0:
                        accuracy_loss = abs(orig_val - dec_val) / abs(orig_val) * 100
                    else:
                        accuracy_loss = abs(dec_val) if dec_val != 0 else 0
                except Exception:
                    accuracy_loss = 100.0  # If query fails, max loss

                total_latency = encryption_time + decryption_time + query_time

                results.append({
                    "iteration": i+1,
                    "query": query,
                    "encryption_time": encryption_time,
                    "decryption_time": decryption_time,
                    "query_time": query_time,
                    "latency": total_latency,
                    "cpu_usage": psutil.cpu_percent(interval=0.05),
                    "accuracy_loss_percent": accuracy_loss,
                    "success": 1 if accuracy_loss < 100 else 0
                })

            conn_orig.close()
            conn_dec.close()
        return results

    def _compute_summary(self, results):
        df = pd.DataFrame(results)
        summary = {
            'latency': {
                'average': df['encryption_time'].mean() + df['decryption_time'].mean(),
                'median': (df['encryption_time'] + df['decryption_time']).median(),
                'min': (df['encryption_time'] + df['decryption_time']).min(),
                'max': (df['encryption_time'] + df['decryption_time']).max(),
            },
            'cpu': {
                'average': df['cpu_usage'].mean(),
                'median': df['cpu_usage'].median(),
                'max': df['cpu_usage'].max(),
            },
            'utility': {
                'avg_accuracy_loss_percent': df['accuracy_loss_percent'].mean(),
                'median_accuracy_loss_percent': df['accuracy_loss_percent'].median(),
                'min_accuracy_loss_percent': df['accuracy_loss_percent'].min(),
                'max_accuracy_loss_percent': df['accuracy_loss_percent'].max(),
                'std_accuracy_loss_percent': df['accuracy_loss_percent'].std(),
                # 'avg_accuracy_loss_percent': 100.0 - df['accuracy_loss_percent'].mean(),
                # 'median_accuracy_loss_percent': 100.0 - df['accuracy_loss_percent'].median(),
                # 'min_accuracy_loss_percent': 100.0 - df['accuracy_loss_percent'].max(),
                # 'max_accuracy_loss_percent': 100.0 - df['accuracy_loss_percent'].min(),
                # 'std_accuracy_loss_percent': df['accuracy_loss_percent'].std(),
            },
            'overall': {
                # 'success_rate': 100.0 if df['accuracy_loss_percent'].min() == 100.0 else 0.0,
                'success_rate': 100.0 * df['success'].mean(),
                'total_iterations': len(df),
                'queries_per_iteration': len(df),
                'throughput': len(df) / (df['encryption_time'].sum() + df['decryption_time'].sum()) if (df['encryption_time'].sum() + df['decryption_time'].sum()) > 0 else 0,
            }
        }
        return summary
    
    def generate_report(self, results):
        summary = self._compute_summary(results)
        report = []
        report.append("AES ENCRYPTION SYSTEM - PERFORMANCE ASSESSMENT REPORT")
        report.append("=" * 65)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Tested Columns: {', '.join(self.columns)}")
        report.append(f"Number of Iterations: {summary['overall']['total_iterations']}")
        report.append("")

        # Detailed per-iteration performance
        # report.append("🔍 DETAILED ITERATION PERFORMANCE")
        # report.append("=" * 55)
        # for r in results:
        #     report.append(f"Iteration {r['iteration']}:")
        #     report.append(f"  Encryption Time: {r['encryption_time']:.4f} s")
        #     report.append(f"  Decryption Time: {r['decryption_time']:.4f} s")
        #     report.append(f"  Total Latency:   {(r['encryption_time'] + r['decryption_time']):.4f} s")
        #     report.append(f"  Utility (match %): {r['accuracy_loss_percent']:.2f}%")
        #     report.append(f"  CPU Usage: {r['cpu_usage']:.2f}%")
        #     report.append("")

        report.append("🔍 DETAILED QUERY-BY-QUERY PERFORMANCE")
        report.append("=" * 55)
        for idx, r in enumerate(results, 1):
            query_preview = r['query'][:60] + "..." if len(r['query']) > 60 else r['query']
            report.append(f"Query {idx} (Iteration {r['iteration']}):")
            report.append(f"  Encryption Time: {r['encryption_time']:.4f} s")
            report.append(f"  Decryption Time: {r['decryption_time']:.4f} s")
            report.append(f"  Query Time:      {r['query_time']:.4f} s")
            report.append(f"  Total Latency:   {r['latency']:.4f} s")
            report.append(f"  Utility (accuracy loss %): {r['accuracy_loss_percent']:.2f}%")
            report.append(f"  CPU Usage: {r['cpu_usage']:.2f}%")
            report.append(f"  SQL: {query_preview}")
            report.append("")

        report.append("=" * 65)
        report.append("")

        # Latency
        report.append("🕐 LATENCY METRICS")
        report.append("-" * 20)
        report.append(f"Average Latency: {summary['latency']['average']:.4f} seconds")
        report.append(f"Median Latency:  {summary['latency']['median']:.4f} seconds")
        report.append(f"Min Latency:     {summary['latency']['min']:.4f} seconds")
        report.append(f"Max Latency:     {summary['latency']['max']:.4f} seconds")
        report.append("")

        # CPU
        report.append("💻 CPU UTILIZATION METRICS")
        report.append("-" * 27)
        report.append(f"Average CPU Usage: {summary['cpu']['average']:.2f}%")
        report.append(f"Median CPU Usage:  {summary['cpu']['median']:.2f}%")
        report.append(f"Peak CPU Usage:    {summary['cpu']['max']:.2f}%")
        report.append("")

        # Privacy (not applicable)
        report.append("🔒 PRIVACY LEVEL METRICS")
        report.append("-" * 25)
        report.append("Mechanisms Used:")
        report.append("  - AES Encryption: All columns specified")
        report.append("")

        # Utility
        report.append("📊 OVERALL UTILITY METRICS (Accuracy Loss)")
        report.append("-" * 42)
        report.append(f"Average Accuracy Loss:    {summary['utility']['avg_accuracy_loss_percent']:.2f}%")
        report.append(f"Median Accuracy Loss:     {summary['utility']['median_accuracy_loss_percent']:.2f}%")
        report.append(f"Min Accuracy Loss:        {summary['utility']['min_accuracy_loss_percent']:.2f}%")
        report.append(f"Max Accuracy Loss:        {summary['utility']['max_accuracy_loss_percent']:.2f}%")
        report.append(f"Std Dev Accuracy Loss:    {summary['utility']['std_accuracy_loss_percent']:.2f}%")
        report.append("")

        # Overall
        report.append("⚡ OVERALL PERFORMANCE")
        report.append("-" * 22)
        report.append(f"Success Rate:        {summary['overall']['success_rate']:.1f}%")
        report.append(f"Queries Per Iteration: {summary['overall']['queries_per_iteration']}")
        report.append(f"Total Iterations:    {summary['overall']['total_iterations']}")
        report.append(f"Throughput:          {summary['overall']['throughput']:.2f} queries/second")
        report.append("")
        report.append("=" * 65)
        return "\n".join(report)
    
from src.k_anonymity_system import KAnonymitySystem

class KAnonymityPerformanceAssessment:
    """Performance assessment for K-Anonymity system."""
    def __init__(self, csv_path="dataset/adult.csv", quasi_identifiers=None, k=5, num_rows=GLOBAL_NUM_ROWS):
        if quasi_identifiers is None:
            quasi_identifiers = ['age', 'education', 'marital.status']  # Example QIs
        self.csv_path = csv_path
        self.quasi_identifiers = quasi_identifiers
        self.k = k
        self.num_rows = num_rows

    def run_performance_test(self, num_iterations=3):
        import time, psutil
        results = []
        for i in range(num_iterations):
            df = pd.read_csv(self.csv_path).head(self.num_rows)
            ksys = KAnonymitySystem(k=self.k)
            start_time = time.time()
            anonymized_df = ksys.anonymize(df, self.quasi_identifiers)
            latency = time.time() - start_time
            cpu = psutil.cpu_percent(interval=0.1)
            # Utility: percent of rows retained (higher is better)
            utility = 100.0 * len(anonymized_df) / len(df) if len(df) > 0 else 0.0
            results.append({
                "iteration": i+1,
                "latency": latency,
                "cpu_usage": cpu,
                "utility_percent": utility,
                "rows_retained": len(anonymized_df),
                "rows_original": len(df)
            })
        return results

    def _compute_summary(self, results):
        df = pd.DataFrame(results)
        summary = {
            'latency': {
                'average': df['latency'].mean(),
                'median': df['latency'].median(),
                'min': df['latency'].min(),
                'max': df['latency'].max(),
            },
            'cpu': {
                'average': df['cpu_usage'].mean(),
                'median': df['cpu_usage'].median(),
                'max': df['cpu_usage'].max(),
            },
            'utility': {
                'avg_accuracy_loss_percent': 100.0 - df['utility_percent'].mean(),
                'median_accuracy_loss_percent': 100.0 - df['utility_percent'].median(),
                'min_accuracy_loss_percent': 100.0 - df['utility_percent'].max(),
                'max_accuracy_loss_percent': 100.0 - df['utility_percent'].min(),
                'std_accuracy_loss_percent': df['utility_percent'].std(),
            },
            'overall': {
                'success_rate': 100.0 if df['utility_percent'].min() > 0 else 0.0,
                'total_iterations': len(df),
                'queries_per_iteration': len(df),
                'throughput': len(df) / df['latency'].sum() if df['latency'].sum() > 0 else 0,
            }
        }
        return summary

    def generate_report(self, results):
        summary = self._compute_summary(results)
        report = []
        report.append("K-ANONYMITY SYSTEM - PERFORMANCE ASSESSMENT REPORT")
        report.append("=" * 65)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Quasi-Identifiers: {', '.join(self.quasi_identifiers)}")
        report.append(f"K Value: {self.k}")
        report.append(f"Number of Iterations: {summary['overall']['total_iterations']}")
        report.append("")

        # Detailed per-iteration performance
        # report.append("🔍 DETAILED ITERATION PERFORMANCE")
        # report.append("=" * 55)
        # for r in results:
        #     report.append(f"Iteration {r['iteration']}:")
        #     report.append(f"  Latency: {r['latency']:.4f} s")
        #     report.append(f"  Utility (rows retained %): {r['utility_percent']:.2f}%")
        #     report.append(f"  Rows Retained: {r['rows_retained']} / {r['rows_original']}")
        #     report.append(f"  CPU Usage: {r['cpu_usage']:.2f}%")
        #     report.append("")

        report.append("🔍 DETAILED QUERY-BY-QUERY PERFORMANCE")
        report.append("=" * 55)
        for idx, r in enumerate(results, 1):
            report.append(f"Query {idx} (Iteration {r['iteration']}):")
            report.append(f"  Latency: {r['latency']:.4f} s")
            report.append(f"  Utility (rows retained %): {r['utility_percent']:.2f}%")
            report.append(f"  Rows Retained: {r['rows_retained']} / {r['rows_original']}")
            report.append(f"  CPU Usage: {r['cpu_usage']:.2f}%")
            report.append("")

        report.append("=" * 65)
        report.append("")

        # Latency
        report.append("🕐 LATENCY METRICS")
        report.append("-" * 20)
        report.append(f"Average Latency: {summary['latency']['average']:.4f} seconds")
        report.append(f"Median Latency:  {summary['latency']['median']:.4f} seconds")
        report.append(f"Min Latency:     {summary['latency']['min']:.4f} seconds")
        report.append(f"Max Latency:     {summary['latency']['max']:.4f} seconds")
        report.append("")

        # CPU
        report.append("💻 CPU UTILIZATION METRICS")
        report.append("-" * 27)
        report.append(f"Average CPU Usage: {summary['cpu']['average']:.2f}%")
        report.append(f"Median CPU Usage:  {summary['cpu']['median']:.2f}%")
        report.append(f"Peak CPU Usage:    {summary['cpu']['max']:.2f}%")
        report.append("")

        # Privacy (K-anonymity)
        report.append("🔒 PRIVACY LEVEL METRICS")
        report.append("-" * 25)
        report.append(f"K Value:                  {self.k}")
        report.append(f"Quasi-Identifiers:        {', '.join(self.quasi_identifiers)}")
        report.append("Mechanisms Used:")
        report.append("  - K-Anonymity Generalization/Suppression")
        report.append("")

        # Utility
        report.append("📊 OVERALL UTILITY METRICS (Accuracy Loss)")
        report.append("-" * 42)
        report.append(f"Average Accuracy Loss:    {summary['utility']['avg_accuracy_loss_percent']:.2f}%")
        report.append(f"Median Accuracy Loss:     {summary['utility']['median_accuracy_loss_percent']:.2f}%")
        report.append(f"Min Accuracy Loss:        {summary['utility']['min_accuracy_loss_percent']:.2f}%")
        report.append(f"Max Accuracy Loss:        {summary['utility']['max_accuracy_loss_percent']:.2f}%")
        report.append(f"Std Dev Accuracy Loss:    {summary['utility']['std_accuracy_loss_percent']:.2f}%")
        report.append("")

        # Overall
        report.append("⚡ OVERALL PERFORMANCE")
        report.append("-" * 22)
        report.append(f"Success Rate:        {summary['overall']['success_rate']:.1f}%")
        report.append(f"Queries Per Iteration: {summary['overall']['queries_per_iteration']}")
        report.append(f"Total Iterations:    {summary['overall']['total_iterations']}")
        report.append(f"Throughput:          {summary['overall']['throughput']:.2f} queries/second")
        report.append("")
        report.append("=" * 65)
        return "\n".join(report)

# class PerformanceAssessment:
#     """Comprehensive performance assessment for differential privacy system."""
    
#     def __init__(self, dp_system):
#         self.dp_system = dp_system
#         # Use the exact same queries from dp_system.py
#         self.test_queries = DEFAULT_TEST_QUERIES.copy()
#         self.metrics = {
#             'latency': [],
#             'cpu_utilization': [],
#             'privacy_level': [],
#             'utility': []
#         }
#         self.query_results = []
        
#         print(f"📊 Performance Assessment initialized with {len(self.test_queries)} queries from dp_system.py")
#         for i, query in enumerate(self.test_queries, 1):
#             print(f"   {i}. {query}")
        
#     def run_performance_test(self, num_iterations: int = 5) -> Dict[str, Any]:
#         """Run performance test using the same queries as dp_system.py."""
#         print(f"\n🚀 Running Performance Assessment")
#         print(f"🔄 Running {num_iterations} iterations for statistical accuracy...")
        
#         # Clear previous query logs for clean testing
#         self._cleanup_previous_results()
        
#         iteration_results = []
        
#         for iteration in range(num_iterations):
#             print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
#             # Initialize fresh system for each iteration
#             test_system = DifferentialPrivacySystem()
#             test_system.initialize_system()

#             # Reset budget for this iteration
#             self._reset_analyst_budget(test_system, 'analyst_1')
            
#             # Track query IDs for this iteration
#             start_time = time.time()
            
#             # Submit queries and track their IDs
#             iteration_query_ids = []
#             for query in self.test_queries:
#                 query_id = test_system.submit_query(query, 'analyst_1')
#                 iteration_query_ids.append(query_id)
            
#             print(f"Submitted queries: {', '.join(map(str, iteration_query_ids))}")
            
#             # Process with DP
#             test_system.process_queries('analyst_1')
#             test_system.process_batches()
            
#             # Get results for ONLY this iteration's queries
#             results = self._get_iteration_specific_results(test_system, iteration_query_ids)
#             end_time = time.time()
            
#             print(f"📊 Iteration {iteration + 1} Results ({len(results)} queries):")
#             if not results.empty:
#                 print(results[['query_id', 'query_type', 'mechanism', 'noisy_result', 'epsilon_charge', 'status']].to_string(index=False))
#             else:
#                 print("No results found")
            
#             # Analyze results
#             iteration_result = self._analyze_iteration_results(
#                 results, iteration + 1, start_time, end_time
#             )
#             iteration_results.append(iteration_result)
        
#         # Compile final results
#         return self._compile_final_results(iteration_results)
    
#     def _reset_analyst_budget(self, test_system, analyst_id: str, total_budget: float = 1.0):
#         """Reset budget for analyst to fresh state."""
#         try:
#             with test_system.db_manager.get_connection() as conn:
#                 conn.execute(text("""
#                     UPDATE analyst_budget 
#                     SET epsilon_spent = 0.0, epsilon_total = :epsilon_total 
#                     WHERE analyst_id = :analyst_id
#                 """), {
#                     'analyst_id': analyst_id,
#                     'epsilon_total': total_budget
#                 })
#                 conn.commit()
#                 print(f"🔄 Reset budget to {total_budget} epsilon for {analyst_id}")
#         except Exception as e:
#             print(f"Warning: Could not reset budget for {analyst_id}: {e}")

#     def _cleanup_previous_results(self):
#         """Clear previous query logs for clean testing."""
#         try:
#             with self.dp_system.db_manager.get_connection() as conn:
#                 # Clear ALL previous query logs
#                 conn.execute(text("DELETE FROM dp_query_log"))
#                 conn.execute(text("DELETE FROM dp_batch"))
#                 conn.execute(text("DELETE FROM dp_batch_member"))
#                 conn.execute(text("DELETE FROM dp_measurement"))
                
#                 # Reset privacy budget
#                 conn.execute(text("UPDATE analyst_budget SET epsilon_spent = 0.0 WHERE analyst_id = 'analyst_1'"))
#                 conn.commit()
                
#                 print("🧹 Cleared all previous query logs and reset privacy budget")
#         except Exception as e:
#             print(f"Warning: Could not clear previous logs: {e}")
    
#     def _get_iteration_specific_results(self, test_system, query_ids: List[int]) -> pd.DataFrame:
#         """Get results for specific query IDs only."""
#         if not query_ids:
#             return pd.DataFrame()
        
#         query_ids_str = ','.join(map(str, query_ids))
#         with test_system.db_manager.get_connection() as conn:
#             return pd.read_sql(
#                 f"SELECT * FROM dp_query_log WHERE query_id IN ({query_ids_str}) ORDER BY query_id", 
#                 conn
#             )
    
#     def _analyze_iteration_results(self, results_df: pd.DataFrame, iteration: int, start_time: float, end_time: float) -> Dict[str, Any]:
#         """Analyze results from a single iteration."""
#         total_latency = end_time - start_time
        
#         # Process each query result
#         query_results = []
#         print(f"\n🔍 Detailed Accuracy Loss Analysis for Iteration {iteration}:")
#         print("-" * 60)
        
#         for _, row in results_df.iterrows():
#             accuracy_loss = self._calculate_utility_score(row)
#             true_result = self._get_true_result(row['raw_sql'])
            
#             # Display individual query accuracy loss
#             query_summary = row['raw_sql'][:50] + "..." if len(row['raw_sql']) > 50 else row['raw_sql']
#             print(f"Query {row['query_id']} ({row['query_type']}): {accuracy_loss:.2f}% accuracy loss")
#             print(f"  SQL: {query_summary}")
#             print(f"  True Result: {true_result:.4f}, Noisy Result: {row['noisy_result']:.4f}")
#             print(f"  Mechanism: {row['mechanism']}, Epsilon: {row['epsilon_charge']}")
#             print()
            
#             query_result = {
#                 'iteration': iteration,
#                 'query_id': row['query_id'],
#                 'query_text': row['raw_sql'],
#                 'query_type': row['query_type'],
#                 'latency': total_latency / len(results_df) if len(results_df) > 0 else total_latency,
#                 'cpu_usage': psutil.cpu_percent(),
#                 'accuracy_loss_percent': accuracy_loss,
#                 'epsilon_used': row['epsilon_charge'],
#                 'mechanism': row['mechanism'],
#                 'noisy_result': row['noisy_result'],
#                 'true_result': true_result,  # Add true result
#                 'status': row['status']
#             }
#             query_results.append(query_result)
        
#         return {
#             'iteration': iteration,
#             'total_latency': total_latency,
#             'query_results': query_results,
#             'summary': {
#                 'avg_latency': total_latency / len(results_df) if len(results_df) > 0 else 0,
#                 'total_epsilon': results_df['epsilon_charge'].sum() if not results_df.empty else 0,
#                 'success_rate': len(results_df[results_df['status'] == 'DONE']) / len(results_df) * 100 if not results_df.empty else 0
#             }
#         }

#     def _get_true_result(self, raw_sql: str) -> float:
#         """Get the true result for a query."""
#         try:
#             with self.dp_system.db_manager.get_connection() as conn:
#                 result = conn.execute(text(raw_sql))
#                 row = result.fetchone()
#                 return float(row[0]) if row and row[0] is not None else 0
#         except Exception as e:
#             print(f"Warning: Could not get true result for query: {e}")
#             return 0
    
#     def _calculate_utility_score(self, query_result) -> float:
#         """Calculate accuracy loss (%) for a query result by comparing noisy vs true result."""
#         try:
#             # Get true result by executing the raw SQL
#             true_value = self._get_true_result(query_result['raw_sql'])
            
#             # Get noisy result
#             noisy_value = float(query_result['noisy_result']) if pd.notna(query_result['noisy_result']) else 0
            
#             # Calculate accuracy loss percentage
#             if true_value != 0:
#                 accuracy_loss = abs(true_value - noisy_value) / abs(true_value) * 100
#             else:
#                 # Handle case where true value is 0
#                 accuracy_loss = abs(noisy_value) if noisy_value != 0 else 0

#             return accuracy_loss
#         except Exception as e:
#             print(f"Warning: Could not calculate accuracy loss for query {query_result.get('query_id', 'unknown')}: {e}")
#             return float('inf')
    
#     def _compile_final_results(self, iteration_results: List[Dict]) -> Dict[str, Any]:
#         """Compile final results from all iterations."""
#         all_query_results = []
#         for iteration_result in iteration_results:
#             all_query_results.extend(iteration_result['query_results'])
        
#         # Calculate summary statistics
#         if all_query_results:
#             df = pd.DataFrame(all_query_results)
            
#             # Filter out infinite values for accuracy loss calculations
#             valid_accuracy_loss = df[df['accuracy_loss_percent'] != float('inf')]['accuracy_loss_percent']
            
#             # Calculate per-iteration epsilon usage
#             num_iterations = len(iteration_results)
#             epsilon_per_iteration = df['epsilon_used'].sum() / num_iterations if num_iterations > 0 else 0

#             summary = {
#                 'latency': {
#                     'average': df['latency'].mean(),
#                     'median': df['latency'].median(),
#                     'std': df['latency'].std(),
#                     'min': df['latency'].min(),
#                     'max': df['latency'].max()
#                 },
#                 'cpu': {
#                     'average': df['cpu_usage'].mean(),
#                     'median': df['cpu_usage'].median(),
#                     'max': df['cpu_usage'].max()
#                 },
#                 'utility': {  # Now represents accuracy loss
#                     'avg_accuracy_loss_percent': valid_accuracy_loss.mean() if not valid_accuracy_loss.empty else 0,
#                     'median_accuracy_loss_percent': valid_accuracy_loss.median() if not valid_accuracy_loss.empty else 0,
#                     'min_accuracy_loss_percent': valid_accuracy_loss.min() if not valid_accuracy_loss.empty else 0,
#                     'max_accuracy_loss_percent': valid_accuracy_loss.max() if not valid_accuracy_loss.empty else 0,
#                     'std_accuracy_loss_percent': valid_accuracy_loss.std() if not valid_accuracy_loss.empty else 0
#                 },
#                 'privacy': {
#                     'epsilon_per_iteration': epsilon_per_iteration, 
#                     'total_iterations': num_iterations,  
#                     'total_epsilon_across_iterations': df['epsilon_used'].sum(),
#                     'avg_epsilon_per_query': df['epsilon_used'].mean(),
#                     'mechanisms_used': df['mechanism'].value_counts().to_dict()
#                 },
#                 'overall': {
#                     'total_queries': len(df),
#                     'queries_per_iteration': len(df) // num_iterations if num_iterations > 0 else 0,
#                     'success_rate': len(df[df['status'] == 'DONE']) / len(df) * 100,
#                     'throughput': len(df) / df['latency'].sum() if df['latency'].sum() > 0 else 0
#                 }
#             }
#         else:
#             summary = {}
        
#         return {
#             'query_results': all_query_results,
#             'summary': summary,
#             'test_queries': self.test_queries,
#             'timestamp': datetime.now().isoformat()
#         }
    
#     def _generate_performance_report(self, results: List[Dict], summary: Dict) -> str:
#         """Generate a detailed performance report."""
        
#         report = []
#         report.append("DIFFERENTIAL PRIVACY SYSTEM - PERFORMANCE ASSESSMENT REPORT")
#         report.append("=" * 65)
#         report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#         report.append(f"Test Queries from Differential Privacy System: {len(self.test_queries)}")
#         report.append("")
        
#         # Add detailed per-query accuracy loss analysis
#         if results:
#             report.append("🔍 DETAILED QUERY-BY-QUERY ACCURACY LOSS ANALYSIS")
#             report.append("=" * 55)
            
#             # Group results by query type for better analysis
#             df = pd.DataFrame(results)
            
#             # Analyze by query type
#             for query_type in ['single', 'mean', 'batch']:
#                 type_results = df[df['query_type'] == query_type]
#                 if not type_results.empty:
#                     report.append(f"\n📊 {query_type.upper()} QUERIES:")
#                     report.append("-" * 20)
                    
#                     for _, row in type_results.iterrows():
#                         if row['accuracy_loss_percent'] != float('inf'):
#                             report.append(f"Query {row['query_id']} (Iteration {row['iteration']}):")
#                             report.append(f"  True Result:    {row['true_result']:.4f}")
#                             report.append(f"  Noisy Result:   {row['noisy_result']:.4f}")
#                             report.append(f"  Accuracy Loss:  {row['accuracy_loss_percent']:.2f}%")
#                             report.append(f"  Mechanism:      {row['mechanism']}")
#                             report.append(f"  Epsilon Used:   {row['epsilon_used']:.2f}")
#                             query_preview = row['query_text'][:60] + "..." if len(row['query_text']) > 60 else row['query_text']
#                             report.append(f"  SQL:            {query_preview}")
#                             report.append("")
            
#             # Summary statistics by query type
#             report.append("\n📈 ACCURACY LOSS SUMMARY BY QUERY TYPE:")
#             report.append("-" * 45)
            
#             for query_type in ['single', 'mean', 'batch']:
#                 type_results = df[df['query_type'] == query_type]
#                 valid_results = type_results[type_results['accuracy_loss_percent'] != float('inf')]
                
#                 if not valid_results.empty:
#                     avg_loss = valid_results['accuracy_loss_percent'].mean()
#                     median_loss = valid_results['accuracy_loss_percent'].median()
#                     min_loss = valid_results['accuracy_loss_percent'].min()
#                     max_loss = valid_results['accuracy_loss_percent'].max()
                    
#                     report.append(f"{query_type.upper()} Queries ({len(valid_results)} queries):")
#                     report.append(f"  Average Loss: {avg_loss:.2f}%")
#                     report.append(f"  Median Loss:  {median_loss:.2f}%")
#                     report.append(f"  Range:        {min_loss:.2f}% - {max_loss:.2f}%")
#                     report.append("")
            
#             report.append("=" * 65)
#             report.append("")
        
#         if summary:
#             # Latency Report
#             report.append("🕐 LATENCY METRICS")
#             report.append("-" * 20)
#             report.append(f"Average Latency: {summary['latency']['average']:.4f} seconds")
#             report.append(f"Median Latency:  {summary['latency']['median']:.4f} seconds")
#             report.append(f"Min Latency:     {summary['latency']['min']:.4f} seconds")
#             report.append(f"Max Latency:     {summary['latency']['max']:.4f} seconds")
#             report.append("")
            
#             # CPU Utilization Report
#             report.append("💻 CPU UTILIZATION METRICS")
#             report.append("-" * 27)
#             report.append(f"Average CPU Usage: {summary['cpu']['average']:.2f}%")
#             report.append(f"Median CPU Usage:  {summary['cpu']['median']:.2f}%")
#             report.append(f"Peak CPU Usage:    {summary['cpu']['max']:.2f}%")
#             report.append("")
            
#             # Privacy Report
#             report.append("🔒 PRIVACY LEVEL METRICS")
#             report.append("-" * 25)
#             report.append(f"Epsilon Used Per Iteration:  {summary['privacy']['epsilon_per_iteration']:.4f}")
#             report.append(f"Number of Iterations:        {summary['privacy']['total_iterations']}")
#             report.append(f"Total Budget Per Iteration:  1.0000 (reset each iteration)")
#             report.append(f"Budget Utilization:          {summary['privacy']['epsilon_per_iteration']/1.0*100:.1f}%")
#             report.append(f"Average Epsilon/Query:       {summary['privacy']['avg_epsilon_per_query']:.4f}")
#             report.append("")
#             report.append("Mechanisms Used:")
#             for mechanism, count in summary['privacy']['mechanisms_used'].items():
#                 report.append(f"  - {mechanism}: {count} queries")
#             report.append("")
            
#             # Utility Report - UPDATED FOR ACCURACY LOSS
#             report.append("📊 OVERALL UTILITY METRICS (Accuracy Loss)")
#             report.append("-" * 42)
#             report.append(f"Average Accuracy Loss:    {summary['utility']['avg_accuracy_loss_percent']:.2f}%")
#             report.append(f"Median Accuracy Loss:     {summary['utility']['median_accuracy_loss_percent']:.2f}%")
#             report.append(f"Min Accuracy Loss:        {summary['utility']['min_accuracy_loss_percent']:.2f}%")
#             report.append(f"Max Accuracy Loss:        {summary['utility']['max_accuracy_loss_percent']:.2f}%")
#             report.append(f"Std Dev Accuracy Loss:    {summary['utility']['std_accuracy_loss_percent']:.2f}%")
#             report.append("")
            
#             # Overall Performance
#             report.append("⚡ OVERALL PERFORMANCE")
#             report.append("-" * 22)
#             report.append(f"Success Rate:        {summary['overall']['success_rate']:.1f}%")
#             report.append(f"Queries Per Iteration: {summary['overall']['queries_per_iteration']}")
#             report.append(f"Total Iterations:    {summary['privacy']['total_iterations']}")
#             report.append(f"Throughput:          {summary['overall']['throughput']:.2f} queries/second")
#             report.append("")
        
#         report.append("=" * 65)
        
#         return "\n".join(report)


# def run_comprehensive_performance_test():
#     """Run comprehensive performance test using dp_system.py queries."""
#     print("🎯 Comprehensive Performance Assessment")
#     print("=" * 60)
    
#     # Create DP system
#     dp_system = DifferentialPrivacySystem()
    
#     # Create performance assessor
#     assessor = PerformanceAssessment(dp_system)
    
#     # Run performance test with the same queries from dp_system.py
#     results = assessor.run_performance_test(num_iterations=3)
    
#     # Generate report
#     report = assessor._generate_performance_report(
#         results.get('query_results', []), 
#         results.get('summary', {})
#     )
    
#     print(report)
    
#     # Save results
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"performance_results_{timestamp}.json"
    
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(results, f, indent=2, default=str)
    
#     print(f"\n💾 Results saved to: {filename}")
#     return results, filename

class PerformanceAssessment:
    """Comprehensive performance assessment for differential privacy system."""
    
    def __init__(self, dp_system):
        self.dp_system = dp_system
        # Use the exact same queries from dp_system.py
        self.test_queries = DEFAULT_TEST_QUERIES.copy()
        self.metrics = {
            'latency': [],
            'cpu_utilization': [],
            'privacy_level': [],
            'utility': []
        }
        self.query_results = []
        
        print(f"Performance Assessment initialized with {len(self.test_queries)} queries from dp_system.py")
        for i, query in enumerate(self.test_queries, 1):
            print(f"   {i}. {query}")
        
    def run_performance_test(self, num_iterations: int) -> Dict[str, Any]:
        """Run performance test using the same queries as dp_system.py."""
        print(f"\nRunning Performance Assessment")
        print(f"Running {num_iterations} iterations for statistical accuracy...")
        
        # Clear previous query logs for clean testing
        self._cleanup_previous_results()
        
        iteration_results = []
        
        for iteration in range(num_iterations):
            print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")
            
            # Initialize fresh system for each iteration
            test_system = DifferentialPrivacySystem()
            test_system.initialize_system()

            # Reset budget for this iteration
            self._reset_analyst_budget(test_system, 'analyst_1')
            
            # Track query IDs for this iteration
            start_time = time.time()
            
            # Submit queries and track their IDs
            iteration_query_ids = []
            for query in self.test_queries:
                query_id = test_system.submit_query(query, 'analyst_1')
                iteration_query_ids.append(query_id)
            
            print(f"Submitted queries: {', '.join(map(str, iteration_query_ids))}")
            
            # Process with DP
            test_system.process_queries('analyst_1')
            test_system.process_batches()
            
            # Get results for ONLY this iteration's queries
            results = self._get_iteration_specific_results(test_system, iteration_query_ids)
            end_time = time.time()
            
            print(f"Iteration {iteration + 1} Results ({len(results)} queries):")
            if not results.empty:
                print(results[['query_id', 'query_type', 'mechanism', 'noisy_result', 'epsilon_charge', 'status']].to_string(index=False))
            else:
                print("No results found")
            
            # Analyze results
            iteration_result = self._analyze_iteration_results(
                results, iteration + 1, start_time, end_time
            )
            iteration_results.append(iteration_result)
        
        # Compile final results
        return self._compile_final_results(iteration_results)
    
    def _reset_analyst_budget(self, test_system, analyst_id: str, total_budget: float = 1.0):
        """Reset budget for analyst to fresh state."""
        try:
            with test_system.db_manager.get_connection() as conn:
                conn.execute(text("""
                    UPDATE analyst_budget 
                    SET epsilon_spent = 0.0, epsilon_total = :epsilon_total 
                    WHERE analyst_id = :analyst_id
                """), {
                    'analyst_id': analyst_id,
                    'epsilon_total': total_budget
                })
                conn.commit()
                print(f"Reset budget to {total_budget} epsilon for {analyst_id}")
        except Exception as e:
            print(f"Warning: Could not reset budget for {analyst_id}: {e}")

    def _cleanup_previous_results(self):
        """Clear previous query logs for clean testing."""
        try:
            with self.dp_system.db_manager.get_connection() as conn:
                # Clear ALL previous query logs
                conn.execute(text("DELETE FROM dp_query_log"))
                conn.execute(text("DELETE FROM dp_batch"))
                conn.execute(text("DELETE FROM dp_batch_member"))
                conn.execute(text("DELETE FROM dp_measurement"))
                
                # Reset privacy budget
                conn.execute(text("UPDATE analyst_budget SET epsilon_spent = 0.0 WHERE analyst_id = 'analyst_1'"))
                conn.commit()
                
                print("Cleared all previous query logs and reset privacy budget")
        except Exception as e:
            print(f"Warning: Could not clear previous logs: {e}")
    
    def _get_iteration_specific_results(self, test_system, query_ids: List[int]) -> pd.DataFrame:
        """Get results for specific query IDs only."""
        if not query_ids:
            return pd.DataFrame()
        
        query_ids_str = ','.join(map(str, query_ids))
        with test_system.db_manager.get_connection() as conn:
            return pd.read_sql(
                f"SELECT * FROM dp_query_log WHERE query_id IN ({query_ids_str}) ORDER BY query_id", 
                conn
            )
    
    def _analyze_iteration_results(self, results_df: pd.DataFrame, iteration: int, start_time: float, end_time: float) -> Dict[str, Any]:
        """Analyze results from a single iteration with per-query measurements."""
        
        # Process each query result
        query_results = []
        print(f"\nDetailed Per-Query Performance Analysis for Iteration {iteration}:")
        print("-" * 70)
        
        for _, row in results_df.iterrows():
            # Measure individual query latency and CPU
            query_start_time = time.time()
            cpu_before = psutil.cpu_percent(interval=None)  # Get current CPU usage
            
            # Calculate accuracy loss (this involves executing the true query)
            accuracy_loss = self._calculate_utility_score(row)
            true_result = self._get_true_result(row['raw_sql'])
            
            # Measure after processing
            query_end_time = time.time()
            cpu_after = psutil.cpu_percent(interval=None)
            
            # Calculate individual metrics
            individual_latency = query_end_time - query_start_time
            individual_cpu = max(cpu_after, cpu_before)  # Take the higher reading
            
            # Display individual query performance
            query_summary = row['raw_sql'][:50] + "..." if len(row['raw_sql']) > 50 else row['raw_sql']
            print(f"Query {row['query_id']} ({row['query_type']}):")
            print(f"  Latency:        {individual_latency:.4f} seconds")
            print(f"  CPU Usage:      {individual_cpu:.1f}%")
            print(f"  Accuracy Loss:  {accuracy_loss:.2f}%")
            print(f"  Mechanism:      {row['mechanism']}, Epsilon: {row['epsilon_charge']}")
            print(f"  SQL:            {query_summary}")
            print(f"  Results:        True: {true_result:.4f}, Noisy: {row['noisy_result']:.4f}")
            print()
            
            query_result = {
                'iteration': iteration,
                'query_id': row['query_id'],
                'query_text': row['raw_sql'],
                'query_type': row['query_type'],
                'latency': individual_latency,              # Individual query latency
                'cpu_usage': individual_cpu,                # Individual query CPU usage
                'accuracy_loss_percent': accuracy_loss,
                'epsilon_used': row['epsilon_charge'],
                'mechanism': row['mechanism'],
                'noisy_result': row['noisy_result'],
                'true_result': true_result,
                'status': row['status']
            }
            query_results.append(query_result)
        
        return {
            'iteration': iteration,
            'total_latency': end_time - start_time,         # Total batch time for reference
            'query_results': query_results,                 # Individual measurements
            'summary': {
                'avg_latency': sum(q['latency'] for q in query_results) / len(query_results) if query_results else 0,
                'avg_cpu': sum(q['cpu_usage'] for q in query_results) / len(query_results) if query_results else 0,
                'total_epsilon': results_df['epsilon_charge'].sum() if not results_df.empty else 0,
                'success_rate': len(results_df[results_df['status'] == 'DONE']) / len(results_df) * 100 if not results_df.empty else 0
            }
        }

    def _get_true_result(self, raw_sql: str) -> float:
        """Get the true result for a query."""
        try:
            with self.dp_system.db_manager.get_connection() as conn:
                result = conn.execute(text(raw_sql))
                row = result.fetchone()
                return float(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            print(f"Warning: Could not get true result for query: {e}")
            return 0
    
    def _calculate_utility_score(self, query_result) -> float:
        """Calculate accuracy loss (%) for a query result by comparing noisy vs true result."""
        try:
            # Get true result by executing the raw SQL
            true_value = self._get_true_result(query_result['raw_sql'])
            
            # Get noisy result
            noisy_value = float(query_result['noisy_result']) if pd.notna(query_result['noisy_result']) else 0
            
            # Calculate accuracy loss percentage
            if true_value != 0:
                accuracy_loss = abs(true_value - noisy_value) / abs(true_value) * 100
            else:
                # Handle case where true value is 0
                accuracy_loss = abs(noisy_value) if noisy_value != 0 else 0

            return accuracy_loss
        except Exception as e:
            print(f"Warning: Could not calculate accuracy loss for query {query_result.get('query_id', 'unknown')}: {e}")
            return float('inf')
    
    def _compile_final_results(self, iteration_results: List[Dict]) -> Dict[str, Any]:
        """Compile final results from all iterations with per-query metrics."""
        all_query_results = []
        for iteration_result in iteration_results:
            all_query_results.extend(iteration_result['query_results'])
        
        # Calculate summary statistics
        if all_query_results:
            df = pd.DataFrame(all_query_results)
            
            # Filter out infinite values for accuracy loss calculations
            valid_accuracy_loss = df[df['accuracy_loss_percent'] != float('inf')]['accuracy_loss_percent']
            
            # Calculate per-iteration epsilon usage
            num_iterations = len(iteration_results)
            epsilon_per_iteration = df['epsilon_used'].sum() / num_iterations if num_iterations > 0 else 0

            summary = {
                'latency': {  # Now based on individual query measurements
                    'average': df['latency'].mean(),
                    'median': df['latency'].median(),
                    'std': df['latency'].std(),
                    'min': df['latency'].min(),
                    'max': df['latency'].max()
                },
                'cpu': {  # Now based on individual query measurements
                    'average': df['cpu_usage'].mean(),
                    'median': df['cpu_usage'].median(),
                    'std': df['cpu_usage'].std(),
                    'min': df['cpu_usage'].min(),
                    'max': df['cpu_usage'].max()
                },
                'utility': {  # Now represents accuracy loss
                    'avg_accuracy_loss_percent': valid_accuracy_loss.mean() if not valid_accuracy_loss.empty else 0,
                    'median_accuracy_loss_percent': valid_accuracy_loss.median() if not valid_accuracy_loss.empty else 0,
                    'min_accuracy_loss_percent': valid_accuracy_loss.min() if not valid_accuracy_loss.empty else 0,
                    'max_accuracy_loss_percent': valid_accuracy_loss.max() if not valid_accuracy_loss.empty else 0,
                    'std_accuracy_loss_percent': valid_accuracy_loss.std() if not valid_accuracy_loss.empty else 0
                },
                'privacy': {
                    'epsilon_per_iteration': epsilon_per_iteration, 
                    'total_iterations': num_iterations,  
                    'total_epsilon_across_iterations': df['epsilon_used'].sum(),
                    'avg_epsilon_per_query': df['epsilon_used'].mean(),
                    'mechanisms_used': df['mechanism'].value_counts().to_dict()
                },
                'overall': {
                    'total_queries': len(df),
                    'queries_per_iteration': len(df) // num_iterations if num_iterations > 0 else 0,
                    'success_rate': len(df[df['status'] == 'DONE']) / len(df) * 100,
                    'throughput': len(df) / df['latency'].sum() if df['latency'].sum() > 0 else 0
                }
            }
        else:
            summary = {}
        
        return {
            'query_results': all_query_results,
            'summary': summary,
            'test_queries': self.test_queries,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_performance_report(self, results: List[Dict], summary: Dict) -> str:
        """Generate a detailed performance report with per-query metrics."""
        
        report = []
        report.append("DIFFERENTIAL PRIVACY SYSTEM - PERFORMANCE ASSESSMENT REPORT")
        report.append("=" * 65)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Test Queries from Differential Privacy System: {len(self.test_queries)}")
        report.append("")
        
        # Add detailed per-query performance analysis
        if results:
            report.append("DETAILED PER-QUERY PERFORMANCE ANALYSIS")
            report.append("=" * 48)
            
            # Group results by actual SQL query (not query type)
            df = pd.DataFrame(results)
            
            # Group by the actual SQL text to show performance for each unique query
            for i, test_query in enumerate(self.test_queries, 1):
                # Find all results for this specific SQL query
                query_results = df[df['query_text'] == test_query]
                
                if not query_results.empty:
                    # Get first row to determine query type and mechanism
                    first_result = query_results.iloc[0]
                    
                    report.append(f"\nQUERY {i} ({first_result['query_type'].upper()}):")
                    report.append("-" * 30)
                    query_preview = test_query[:80] + "..." if len(test_query) > 80 else test_query
                    report.append(f"SQL: {query_preview}")
                    report.append("")
                    
                    # Show results from each iteration for this query
                    valid_results = query_results[query_results['accuracy_loss_percent'] != float('inf')]
                    
                    if not valid_results.empty:
                        # Calculate statistics for this specific query across iterations
                        avg_latency = valid_results['latency'].mean()
                        avg_cpu = valid_results['cpu_usage'].mean()
                        avg_loss = valid_results['accuracy_loss_percent'].mean()
                        median_latency = valid_results['latency'].median()
                        median_cpu = valid_results['cpu_usage'].median()
                        median_loss = valid_results['accuracy_loss_percent'].median()
                        std_latency = valid_results['latency'].std()
                        std_cpu = valid_results['cpu_usage'].std()
                        std_loss = valid_results['accuracy_loss_percent'].std()
                        
                        # Performance summary for this specific query
                        report.append(f"Performance Across {len(valid_results)} Iterations:")
                        report.append(f"  Latency:        Avg: {avg_latency:.4f}s, Median: {median_latency:.4f}s, Std: {std_latency:.4f}s")
                        report.append(f"  CPU Usage:      Avg: {avg_cpu:.1f}%, Median: {median_cpu:.1f}%, Std: {std_cpu:.1f}%")
                        report.append(f"  Accuracy Loss:  Avg: {avg_loss:.2f}%, Median: {median_loss:.2f}%, Std: {std_loss:.2f}%")
                        report.append(f"  Mechanism:      {first_result['mechanism']}")
                        report.append(f"  Avg Epsilon:    {valid_results['epsilon_used'].mean():.3f}")
                        report.append("")
                        
                        # Show individual iteration results for this query
                        report.append("Individual Iteration Results:")
                        for _, row in valid_results.iterrows():
                            report.append(f"  Iteration {row['iteration']}: "
                                        f"Latency: {row['latency']:.4f}s, "
                                        f"CPU: {row['cpu_usage']:.1f}%, "
                                        f"Loss: {row['accuracy_loss_percent']:.2f}%, "
                                        f"True: {row['true_result']:.2f}, "
                                        f"Noisy: {row['noisy_result']:.2f}")
                        report.append("")
        
        report.append("=" * 65)
        report.append("")
        
        # Overall summary by query type (keep this for comparison)
        report.append("SUMMARY BY QUERY TYPE:")
        report.append("-" * 28)
        
        for query_type in ['single', 'mean', 'batch']:
            type_results = df[df['query_type'] == query_type]
            valid_results = type_results[type_results['accuracy_loss_percent'] != float('inf')]
            
            if not valid_results.empty:
                avg_latency = valid_results['latency'].mean()
                avg_cpu = valid_results['cpu_usage'].mean()
                avg_loss = valid_results['accuracy_loss_percent'].mean()
                
                report.append(f"{query_type.upper()} Queries ({len(valid_results)} measurements):")
                report.append(f"  Avg Latency: {avg_latency:.4f}s, Avg CPU: {avg_cpu:.1f}%, Avg Loss: {avg_loss:.2f}%")
        
        report.append("")
        report.append("=" * 65)
        report.append("")
    
        if summary:
            # Latency Report - Now based on individual measurements
            report.append("OVERALL LATENCY METRICS (Per-Query Measurements)")
            report.append("-" * 54)
            report.append(f"Average Latency: {summary['latency']['average']:.4f} seconds")
            report.append(f"Median Latency:  {summary['latency']['median']:.4f} seconds")
            report.append(f"Min Latency:     {summary['latency']['min']:.4f} seconds")
            report.append(f"Max Latency:     {summary['latency']['max']:.4f} seconds")
            report.append(f"Std Dev Latency: {summary['latency']['std']:.4f} seconds")
            report.append("")
            
            # CPU Utilization Report - Now based on individual measurements
            report.append("OVERALL CPU UTILIZATION METRICS (Per-Query Measurements)")
            report.append("-" * 61)
            report.append(f"Average CPU Usage: {summary['cpu']['average']:.2f}%")
            report.append(f"Median CPU Usage:  {summary['cpu']['median']:.2f}%")
            report.append(f"Min CPU Usage:     {summary['cpu']['min']:.2f}%")
            report.append(f"Peak CPU Usage:    {summary['cpu']['max']:.2f}%")
            report.append(f"Std Dev CPU Usage: {summary['cpu']['std']:.2f}%")
            report.append("")
            
            # Privacy Report
            report.append("PRIVACY LEVEL METRICS")
            report.append("-" * 25)
            report.append(f"Epsilon Used Per Iteration:  {summary['privacy']['epsilon_per_iteration']:.4f}")
            report.append(f"Number of Iterations:        {summary['privacy']['total_iterations']}")
            report.append(f"Total Budget Per Iteration:  1.0000 (reset each iteration)")
            report.append(f"Budget Utilization:          {summary['privacy']['epsilon_per_iteration']/1.0*100:.1f}%")
            report.append(f"Average Epsilon/Query:       {summary['privacy']['avg_epsilon_per_query']:.4f}")
            report.append("")
            report.append("Mechanisms Used:")
            for mechanism, count in summary['privacy']['mechanisms_used'].items():
                report.append(f"  - {mechanism}: {count} queries")
            report.append("")
            
            # Utility Report - UPDATED FOR ACCURACY LOSS
            report.append("OVERALL UTILITY METRICS (Accuracy Loss)")
            report.append("-" * 42)
            report.append(f"Average Accuracy Loss:    {summary['utility']['avg_accuracy_loss_percent']:.2f}%")
            report.append(f"Median Accuracy Loss:     {summary['utility']['median_accuracy_loss_percent']:.2f}%")
            report.append(f"Min Accuracy Loss:        {summary['utility']['min_accuracy_loss_percent']:.2f}%")
            report.append(f"Max Accuracy Loss:        {summary['utility']['max_accuracy_loss_percent']:.2f}%")
            report.append(f"Std Dev Accuracy Loss:    {summary['utility']['std_accuracy_loss_percent']:.2f}%")
            report.append("")
            
            # Overall Performance
            report.append("OVERALL PERFORMANCE")
            report.append("-" * 22)
            report.append(f"Success Rate:        {summary['overall']['success_rate']:.1f}%")
            report.append(f"Queries Per Iteration: {summary['overall']['queries_per_iteration']}")
            report.append(f"Total Iterations:    {summary['privacy']['total_iterations']}")
            report.append(f"Throughput:          {summary['overall']['throughput']:.2f} queries/second")
            report.append("")
        
        report.append("=" * 65)
        return "\n".join(report)


def run_comprehensive_performance_test():
    """Run comprehensive performance test using dp_system.py queries."""
    print("Comprehensive Performance Assessment")
    print("=" * 60)
    
    # Create DP system
    dp_system = DifferentialPrivacySystem()
    
    # Create performance assessor
    assessor = PerformanceAssessment(dp_system)
    
    # Run performance test with the same queries from dp_system.py
    results = assessor.run_performance_test(num_iterations=3)
    
    # Generate report
    report = assessor._generate_performance_report(
        results.get('query_results', []), 
        results.get('summary', {})
    )
    
    print(report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {filename}")
    return results, filename


def run_all_performance_tests():
    # Differential Privacy Assessment
    results_dp, filename_dp = run_comprehensive_performance_test()

    # AES Assessment
    aes_assessor = AESPerformanceAssessment(num_rows=GLOBAL_NUM_ROWS)
    aes_results = aes_assessor.run_performance_test(num_iterations=3)
    aes_report = aes_assessor.generate_report(aes_results)
    print(aes_report)

    # Save AES results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_aes = f"aes_performance_results_{timestamp}.json"
    with open(filename_aes, 'w', encoding='utf-8') as f:
        json.dump(aes_results, f, indent=2, default=str)
    print(f"\n💾 AES Results saved to: {filename_aes}")

    # K-Anonymity Assessment
    k_assessor = KAnonymityPerformanceAssessment(num_rows=GLOBAL_NUM_ROWS)
    k_results = k_assessor.run_performance_test(num_iterations=3)
    k_report = k_assessor.generate_report(k_results)
    print(k_report)

    # Save K-Anonymity results
    filename_k = f"k_anonymity_performance_results_{timestamp}.json"
    with open(filename_k, 'w', encoding='utf-8') as f:
        json.dump(k_results, f, indent=2, default=str)
    print(f"\n💾 K-Anonymity Results saved to: {filename_k}")

if __name__ == "__main__":
    run_all_performance_tests()
