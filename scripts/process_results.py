#!/usr/bin/env python3
"""
Process experiment results from evaluation runs.

This script parses time_taken.txt files from evaluation runs and aggregates
statistics (success rate, average time, etc.) into CSV files for analysis.

Usage:
    python process_results.py --input_dir outputs/eval
    python process_results.py --file outputs/eval/square/20250101_120000/time_taken.txt
"""

import os
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd


# =============================================================================
# Constants
# =============================================================================

REWARD_PATTERNS = [
    r"'reward': array\(\[?([\d\.]+)\]?\)",  # Standard format
    r"'reward': ([\d\.]+)",                  # Alternative format
    r"reward: ([\d\.]+)",                    # Simple format
]

TIME_PATTERNS = [
    r"'time_taken': ([\d\.]+)",  # Standard format
    r"time_taken: ([\d\.]+)",    # Simple format
]


# =============================================================================
# Result Parser
# =============================================================================

class ResultParser:
    """Parses time_taken.txt files to extract experiment results."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def parse(self) -> Optional[Dict[str, Any]]:
        """Parse the file and return statistics."""
        if not os.path.exists(self.file_path):
            print(f"Warning: {self.file_path} does not exist.")
            return None
        
        with open(self.file_path, "r") as f:
            content = f.read()
        
        if not content.strip():
            print(f"Warning: {self.file_path} is empty.")
            return None
        
        rewards, success_times, failure_times = self._extract_data(content)
        
        if not rewards:
            print(f"No valid rewards found in {self.file_path}.")
            return None
        
        return self._compute_statistics(rewards, success_times, failure_times)
    
    def _extract_data(
        self, content: str
    ) -> Tuple[List[float], List[float], List[float]]:
        """Extract rewards and times from file content."""
        rewards = []
        success_times = []
        failure_times = []
        
        for line in content.split("\n"):
            if not line.strip():
                continue
            
            reward = self._match_pattern(line, REWARD_PATTERNS)
            time_taken = self._match_pattern(line, TIME_PATTERNS)
            
            if reward is not None and time_taken is not None:
                rewards.append(reward)
                if reward >= 0.5:  # Success threshold
                    success_times.append(time_taken)
                else:
                    failure_times.append(time_taken)
        
        return rewards, success_times, failure_times
    
    @staticmethod
    def _match_pattern(line: str, patterns: List[str]) -> Optional[float]:
        """Try matching patterns and return first match."""
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
        return None
    
    @staticmethod
    def _compute_statistics(
        rewards: List[float],
        success_times: List[float],
        failure_times: List[float],
    ) -> Dict[str, Any]:
        """Compute statistics from extracted data."""
        total_runs = len(rewards)
        success_count = sum(1 for r in rewards if r >= 0.5)
        failure_count = total_runs - success_count
        
        all_times = success_times + failure_times
        
        return {
            "average_reward": sum(rewards) / total_runs,
            "success_rate": success_count / total_runs,
            "total_runs": total_runs,
            "success_count": success_count,
            "failure_count": failure_count,
            "average_time": sum(all_times) / len(all_times) if all_times else 0,
            "average_success_time": (
                sum(success_times) / len(success_times) if success_times else 0
            ),
            "average_failure_time": (
                sum(failure_times) / len(failure_times) if failure_times else 0
            ),
        }


# =============================================================================
# Results Processor
# =============================================================================

class ExperimentResultsProcessor:
    """Processes experiment results and saves to CSV."""
    
    def __init__(self, output_csv: Optional[str] = None):
        self.output_csv = output_csv
    
    def process_file(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[pd.DataFrame]:
        """Process a single time_taken.txt file."""
        parser = ResultParser(file_path)
        stats = parser.parse()
        
        if stats is None:
            return None
        
        # Build result record
        record = {"folder_path": str(Path(file_path).parent)}
        record.update(stats)
        
        if metadata:
            record.update(metadata)
        
        # Add path-derived metadata
        path_metadata = self._extract_path_metadata(file_path)
        record.update(path_metadata)
        
        df = pd.DataFrame([record])
        
        # Determine output path
        csv_path = self.output_csv or self._determine_csv_path(file_path)
        self._save_to_csv(df, csv_path)
        
        return df
    
    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> pd.DataFrame:
        """Process all time_taken.txt files in a directory."""
        all_results = []
        
        pattern = "**/*.txt" if recursive else "*.txt"
        for txt_file in Path(directory).glob(pattern):
            if txt_file.name == "time_taken.txt":
                result = self.process_file(str(txt_file))
                if result is not None:
                    all_results.append(result)
        
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return pd.DataFrame()
    
    @staticmethod
    def _extract_path_metadata(file_path: str) -> Dict[str, Any]:
        """Extract metadata from file path."""
        parts = Path(file_path).parts
        metadata = {}
        
        # Try to extract environment name
        for i, part in enumerate(parts):
            if part in ["square", "can", "lift", "pusht", "tool_hang", "transport"]:
                metadata["env_name"] = part
                break
        
        # Try to extract timestamp (YYYYMMDD_HHMMSS format)
        for part in parts:
            if re.match(r"\d{8}_\d{6}", part):
                metadata["timestamp"] = part
                break
        
        # Try to extract parameter string
        for part in parts:
            if any(x in part for x in ["TS", "ODE", "SDE", "Euler"]):
                metadata["params"] = part
                break
        
        return metadata
    
    def _determine_csv_path(self, file_path: str) -> str:
        """Determine appropriate CSV output path."""
        path = Path(file_path)
        
        # Find the environment directory
        parts = path.parts
        for i, part in enumerate(parts):
            if part in ["square", "can", "lift", "pusht", "tool_hang", "transport"]:
                env_dir = Path(*parts[: i + 1])
                return str(env_dir / "results.csv")
        
        # Fallback to parent directory
        return str(path.parent / "results.csv")
    
    def _save_to_csv(self, df: pd.DataFrame, csv_path: str) -> None:
        """Save DataFrame to CSV, appending if exists."""
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if csv_path.exists():
            try:
                existing_df = pd.read_csv(csv_path)
                
                # Remove duplicates based on folder_path
                folder_path = df["folder_path"].iloc[0]
                existing_df = existing_df[existing_df["folder_path"] != folder_path]
                
                # Merge columns
                for col in df.columns:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                for col in existing_df.columns:
                    if col not in df.columns:
                        df[col] = None
                
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing CSV: {e}")
        
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process experiment results from evaluation runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a single time_taken.txt file",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path (auto-determined if not specified)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search for time_taken.txt files",
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    processor = ExperimentResultsProcessor(args.output_csv)
    
    if args.file:
        print(f"Processing file: {args.file}")
        processor.process_file(args.file)
    elif args.input_dir:
        print(f"Processing directory: {args.input_dir}")
        processor.process_directory(args.input_dir, args.recursive)
    else:
        print("Error: Please specify --file or --input_dir")
        return 1
    
    print("Processing complete.")
    return 0


if __name__ == "__main__":
    exit(main())

