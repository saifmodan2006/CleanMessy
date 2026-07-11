import pandas as pd
import streamlit as st
import datetime
from typing import List, Dict, Any, Optional

class StateHistory:
    """
    Manages the undo/redo stack and operations history timeline for a Pandas DataFrame.
    """
    def __init__(self, initial_df: pd.DataFrame, file_name: str):
        self.history: List[pd.DataFrame] = [initial_df.copy()]
        self.redo_history: List[pd.DataFrame] = []
        self.actions: List[str] = ["Dataset Loaded"]
        self.redo_actions: List[str] = []
        self.timestamps: List[datetime.datetime] = [datetime.datetime.now()]
        self.redo_timestamps: List[datetime.datetime] = []
        self.file_name: str = file_name
        self.logs: List[str] = [f"[{self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S')}] Loaded dataset: {file_name} with shape {initial_df.shape}"]

    def push_state(self, df: pd.DataFrame, action: str):
        """Pushes a new dataframe state to history and clears the redo stack."""
        self.history.append(df.copy())
        self.actions.append(action)
        self.timestamps.append(datetime.datetime.now())
        self.logs.append(f"[{self.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')}] Applied: {action}. New shape: {df.shape}")
        
        # Clear redo history on new action
        self.redo_history.clear()
        self.redo_actions.clear()
        self.redo_timestamps.clear()

    def undo(self) -> Optional[pd.DataFrame]:
        """Reverts to the previous state, moving the current state to the redo stack."""
        if len(self.history) > 1:
            current_df = self.history.pop()
            current_action = self.actions.pop()
            current_time = self.timestamps.pop()
            
            self.redo_history.append(current_df)
            self.redo_actions.append(current_action)
            self.redo_timestamps.append(current_time)
            
            undone_action = f"Undid: {current_action}"
            self.logs.append(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {undone_action}")
            return self.history[-1].copy()
        return None

    def redo(self) -> Optional[pd.DataFrame]:
        """Re-applies the last undone state, moving it back to the history stack."""
        if len(self.redo_history) > 0:
            next_df = self.redo_history.pop()
            next_action = self.redo_actions.pop()
            next_time = self.redo_timestamps.pop()
            
            self.history.append(next_df)
            self.actions.append(next_action)
            self.timestamps.append(next_time)
            
            redone_action = f"Redid: {next_action}"
            self.logs.append(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {redone_action}")
            return next_df.copy()
        return None

    def get_current_df(self) -> pd.DataFrame:
        """Returns the current active dataframe state."""
        return self.history[-1].copy()

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Returns a list of dicts describing the timeline of operations."""
        timeline = []
        for i in range(len(self.history)):
            timeline.append({
                "index": i,
                "action": self.actions[i],
                "timestamp": self.timestamps[i].strftime("%H:%M:%S"),
                "status": "active" if i == len(self.history) - 1 else "past"
            })
        for i in range(len(self.redo_history) - 1, -1, -1):
            timeline.append({
                "index": len(self.history) + (len(self.redo_history) - 1 - i),
                "action": self.redo_actions[i],
                "timestamp": self.redo_timestamps[i].strftime("%H:%M:%S"),
                "status": "undone"
            })
        return timeline

def init_session_state():
    """Initializes global Streamlit session state properties."""
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "dataset_history" not in st.session_state:
        st.session_state.dataset_history = None  # Will hold StateHistory object
    if "original_df" not in st.session_state:
        st.session_state.original_df = None
    if "current_df" not in st.session_state:
        st.session_state.current_df = None
    if "file_name" not in st.session_state:
        st.session_state.file_name = ""
    if "file_type" not in st.session_state:
        st.session_state.file_type = ""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Home"
    if "auto_save" not in st.session_state:
        st.session_state.auto_save = True
    if "chart_theme" not in st.session_state:
        st.session_state.chart_theme = "plotly_dark"
    if "default_format" not in st.session_state:
        st.session_state.default_format = "CSV"
