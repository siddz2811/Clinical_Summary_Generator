"""
Patient Data Management Module

This module provides functionality to ingest and query patient clinical data
from CSV files, treating them as tables in a relational database.

Tables:
- diagnoses: Medical conditions for patients by episode
- medications: Active medications with frequency and classification
- vitals: Vital sign readings from visits
- notes: Free-text clinical notes
- wounds: Details about active wounds/ulcers
- oasis: Functional assessments (OASIS)
"""

import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any, Union


class PatientDatabase:
    """
    A class to manage patient clinical data from CSV files.
    
    Treats CSV files as relational database tables with patient_id as the
    primary linking key across all tables.
    
    Attributes:
        data_dir (Path): Directory containing the CSV files
        tables (Dict[str, pd.DataFrame]): Dictionary of loaded DataFrames
    """
    
    # Define the expected tables and their primary columns
    TABLE_SCHEMAS = {
        'diagnoses': ['patient_id', 'episode_id', 'diagnosis_description', 'diagnosis_code'],
        'medications': ['patient_id', 'episode_id', 'medication_name', 'frequency', 'classification', 'reason'],
        'vitals': ['patient_id', 'episode_id', 'visit_date', 'vital_type', 'reading', 'min_value', 'max_value'],
        'notes': ['patient_id', 'episode_id', 'note_date', 'note_type', 'note_text'],
        'wounds': ['patient_id', 'episode_id', 'description', 'location', 'onset_date', 'visit_date'],
        'oasis': ['patient_id', 'assessment_date', 'assessment_type', 'grooming', 'bathing', 'toilet_transfer', 'transfer', 'ambulation']
    }
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the PatientDatabase.
        
        Args:
            data_dir: Path to directory containing CSV files (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.tables: Dict[str, pd.DataFrame] = {}
        self._load_all_tables()
    
    def _load_all_tables(self) -> None:
        """Load all CSV files from the data directory into DataFrames."""
        for table_name in self.TABLE_SCHEMAS.keys():
            csv_path = self.data_dir / f"{table_name}.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                # Ensure patient_id is integer type for consistent querying
                if 'patient_id' in df.columns:
                    df['patient_id'] = df['patient_id'].astype(int)
                if 'episode_id' in df.columns:
                    df['episode_id'] = df['episode_id'].astype(int)
                # Parse date columns
                date_columns = [col for col in df.columns if 'date' in col.lower()]
                for col in date_columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                self.tables[table_name] = df
                print(f"Loaded {table_name}: {len(df)} rows")
            else:
                print(f"Warning: {csv_path} not found")
    
    def get_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Get a specific table by name.
        
        Args:
            table_name: Name of the table (diagnoses, medications, vitals, notes, wounds, oasis)
            
        Returns:
            DataFrame if table exists, None otherwise
        """
        return self.tables.get(table_name)
    
    def list_tables(self) -> List[str]:
        """
        List all available tables.
        
        Returns:
            List of table names
        """
        return list(self.tables.keys())
    
    def get_all_patient_ids(self) -> List[int]:
        """
        Get list of all unique patient IDs across all tables.
        
        Returns:
            Sorted list of unique patient IDs
        """
        all_ids = set()
        for df in self.tables.values():
            if 'patient_id' in df.columns:
                all_ids.update(df['patient_id'].unique())
        return sorted(list(all_ids))
    
    def filter_by_patient(
        self, 
        patient_id: int, 
        tables: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter all tables (or specified tables) by patient_id.
        
        Args:
            patient_id: The patient ID to filter by
            tables: Optional list of table names to filter. If None, filters all tables.
            
        Returns:
            Dictionary mapping table names to filtered DataFrames
        """
        tables_to_filter = tables if tables else self.list_tables()
        result = {}
        
        for table_name in tables_to_filter:
            df = self.tables.get(table_name)
            if df is not None and 'patient_id' in df.columns:
                filtered = df[df['patient_id'] == patient_id].copy()
                result[table_name] = filtered
        
        return result
    
    def query_table(
        self,
        table_name: str,
        patient_id: Optional[int] = None,
        episode_id: Optional[int] = None,
        columns: Optional[List[str]] = None,
        **conditions
    ) -> Optional[pd.DataFrame]:
        """
        Query a specific table with optional filters.
        
        Args:
            table_name: Name of the table to query
            patient_id: Optional patient ID filter
            episode_id: Optional episode ID filter
            columns: Optional list of columns to return
            **conditions: Additional column=value conditions
            
        Returns:
            Filtered DataFrame or None if table doesn't exist
            
        Example:
            # Get all vitals for patient 1001 where vital_type is "Pain"
            db.query_table('vitals', patient_id=1001, vital_type='Pain')
        """
        df = self.tables.get(table_name)
        if df is None:
            return None
        
        result = df.copy()
        
        # Apply patient_id filter
        if patient_id is not None and 'patient_id' in result.columns:
            result = result[result['patient_id'] == patient_id]
        
        # Apply episode_id filter
        if episode_id is not None and 'episode_id' in result.columns:
            result = result[result['episode_id'] == episode_id]
        
        # Apply additional conditions
        for col, value in conditions.items():
            if col in result.columns:
                result = result[result[col] == value]
        
        # Select specific columns
        if columns:
            available_cols = [c for c in columns if c in result.columns]
            result = result[available_cols]
        
        return result
    
    def get_patient_summary(self, patient_id: int) -> Dict[str, Any]:
        """
        Get a summary of all data for a specific patient.
        
        Args:
            patient_id: The patient ID to summarize
            
        Returns:
            Dictionary containing summary information for the patient
        """
        patient_data = self.filter_by_patient(patient_id)
        
        summary = {
            'patient_id': patient_id,
            'tables': {}
        }
        
        for table_name, df in patient_data.items():
            table_summary = {
                'row_count': len(df),
                'columns': list(df.columns)
            }
            
            # Add table-specific summaries
            if table_name == 'diagnoses' and not df.empty:
                table_summary['unique_diagnoses'] = df['diagnosis_description'].nunique()
                table_summary['diagnoses_list'] = df['diagnosis_description'].unique().tolist()
            
            elif table_name == 'medications' and not df.empty:
                table_summary['unique_medications'] = df['medication_name'].nunique()
                table_summary['medication_classes'] = df['classification'].unique().tolist()
            
            elif table_name == 'vitals' and not df.empty:
                table_summary['vital_types'] = df['vital_type'].unique().tolist()
                table_summary['date_range'] = {
                    'earliest': df['visit_date'].min(),
                    'latest': df['visit_date'].max()
                }
            
            elif table_name == 'notes' and not df.empty:
                table_summary['note_types'] = df['note_type'].unique().tolist()
                table_summary['date_range'] = {
                    'earliest': df['note_date'].min(),
                    'latest': df['note_date'].max()
                }
            
            elif table_name == 'wounds' and not df.empty:
                table_summary['wound_locations'] = df['location'].unique().tolist()
                table_summary['wound_descriptions'] = df['description'].unique().tolist()
            
            elif table_name == 'oasis' and not df.empty:
                table_summary['assessment_dates'] = df['assessment_date'].unique().tolist()
            
            summary['tables'][table_name] = table_summary
        
        return summary
    
    def get_episodes_for_patient(self, patient_id: int) -> List[int]:
        """
        Get all episode IDs for a specific patient.
        
        Args:
            patient_id: The patient ID to look up
            
        Returns:
            List of unique episode IDs for the patient
        """
        episodes = set()
        for df in self.tables.values():
            if 'patient_id' in df.columns and 'episode_id' in df.columns:
                patient_df = df[df['patient_id'] == patient_id]
                episodes.update(patient_df['episode_id'].unique())
        return sorted(list(episodes))
    
    def join_tables(
        self,
        left_table: str,
        right_table: str,
        on: Union[str, List[str]] = 'patient_id',
        how: str = 'inner',
        patient_id: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Join two tables together, optionally filtered by patient_id.
        
        Args:
            left_table: Name of the left table
            right_table: Name of the right table
            on: Column(s) to join on (default: 'patient_id')
            how: Type of join ('inner', 'left', 'right', 'outer')
            patient_id: Optional patient ID to filter results
            
        Returns:
            Joined DataFrame or None if tables don't exist
        """
        left_df = self.tables.get(left_table)
        right_df = self.tables.get(right_table)
        
        if left_df is None or right_df is None:
            return None
        
        # Apply patient filter before joining if specified
        if patient_id is not None:
            if 'patient_id' in left_df.columns:
                left_df = left_df[left_df['patient_id'] == patient_id]
            if 'patient_id' in right_df.columns:
                right_df = right_df[right_df['patient_id'] == patient_id]
        
        # Perform the join
        result = pd.merge(
            left_df, 
            right_df, 
            on=on, 
            how=how,
            suffixes=('_left', '_right')
        )
        
        return result
    
    def __repr__(self) -> str:
        """String representation of the database."""
        table_info = [f"  - {name}: {len(df)} rows" for name, df in self.tables.items()]
        return f"PatientDatabase(\n" + "\n".join(table_info) + "\n)"


# Convenience function for quick data access
def load_patient_data(data_dir: str = "data") -> PatientDatabase:
    """
    Convenience function to load patient data.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Initialized PatientDatabase instance
    """
    return PatientDatabase(data_dir)


# Example usage and demonstration
if __name__ == "__main__":
    # Initialize the database
    print("=" * 60)
    print("Loading Patient Database")
    print("=" * 60)
    db = PatientDatabase("data")
    
    print("\n" + "=" * 60)
    print("Database Overview")
    print("=" * 60)
    print(db)
    
    # List all patients
    print("\n" + "=" * 60)
    print("Available Patients")
    print("=" * 60)
    patients = db.get_all_patient_ids()
    print(f"Patient IDs: {patients}")
    
    # Demonstrate filtering by patient
    print("\n" + "=" * 60)
    print("Example: Filter by Patient ID 1001")
    print("=" * 60)
    patient_1001_data = db.filter_by_patient(1001)
    for table_name, df in patient_1001_data.items():
        print(f"\n{table_name}: {len(df)} rows")
        if not df.empty:
            print(df.head(3).to_string())
    
    # Demonstrate querying specific table
    print("\n" + "=" * 60)
    print("Example: Query Vitals for Patient 1002")
    print("=" * 60)
    vitals_1002 = db.query_table('vitals', patient_id=1002)
    if vitals_1002 is not None and not vitals_1002.empty:
        print(vitals_1002.head(10).to_string())
    
    # Get patient summary
    print("\n" + "=" * 60)
    print("Example: Patient Summary for Patient 1001")
    print("=" * 60)
    summary = db.get_patient_summary(1001)
    print(f"Patient ID: {summary['patient_id']}")
    for table_name, info in summary['tables'].items():
        print(f"\n{table_name}:")
        print(f"  - Row count: {info['row_count']}")
        if 'unique_diagnoses' in info:
            print(f"  - Unique diagnoses: {info['unique_diagnoses']}")
        if 'unique_medications' in info:
            print(f"  - Unique medications: {info['unique_medications']}")
        if 'vital_types' in info:
            print(f"  - Vital types: {info['vital_types']}")
    
    # Get episodes for patient
    print("\n" + "=" * 60)
    print("Example: Episodes for Each Patient")
    print("=" * 60)
    for pid in patients:
        episodes = db.get_episodes_for_patient(pid)
        print(f"Patient {pid}: Episodes {episodes}")
