"""
Centralized Export Manager for handling both JSON and CSV exports.

This module provides a unified interface for managing automatic exports
based on user settings and ensures consistency between JSON and CSV exports.
"""

import os
from PySide6.QtCore import QSettings
from typing import List, Any, Optional

class ExportManager:
    """Centralized manager for handling automatic exports with user settings."""
    
    def __init__(self, app_instance=None):
        """Initialize the export manager.
        
        Args:
            app_instance: Reference to the main application instance
        """
        self.app = app_instance
        self.settings = QSettings("CalebCoatney", "ChromaKit")
        
    def should_export_json(self, trigger_type: str) -> bool:
        """Check if JSON export should be triggered for the given event type.
        
        Args:
            trigger_type: One of 'integration', 'ms_search', 'assignment', 'batch'
            
        Returns:
            bool: True if JSON export should happen
        """
        # Check if JSON format is enabled at all
        if not self.settings.value("export/json_enabled", True, type=bool):
            return False
            
        # Check trigger-specific settings
        if trigger_type == 'integration':
            return self.settings.value("export/after_integration", True, type=bool)
        elif trigger_type == 'ms_search':
            return self.settings.value("export/after_ms_search", True, type=bool)
        elif trigger_type == 'assignment':
            return self.settings.value("export/after_assignment", True, type=bool)
        elif trigger_type == 'batch':
            return self.settings.value("export/during_batch", True, type=bool)
        return True
        
    def should_export_csv(self, trigger_type: str) -> bool:
        """Check if CSV export should be triggered for the given event type.
        
        Args:
            trigger_type: One of 'integration', 'ms_search', 'assignment', 'batch'
            
        Returns:
            bool: True if CSV export should happen
        """
        # Check if CSV format is enabled at all
        if not self.settings.value("export/csv_enabled", True, type=bool):
            return False
            
        # Check trigger-specific settings (different defaults to maintain current behavior)
        if trigger_type == 'integration':
            return self.settings.value("export/after_integration", False, type=bool)  # Default: OFF
        elif trigger_type == 'ms_search':
            return self.settings.value("export/after_ms_search", True, type=bool)    # Default: ON
        elif trigger_type == 'assignment':
            return self.settings.value("export/after_assignment", False, type=bool)  # Default: OFF
        elif trigger_type == 'batch':
            return self.settings.value("export/during_batch", True, type=bool)       # Default: ON
        return False
        
    def export_results(self, peaks: List[Any], d_path: str, trigger_type: str, 
                      detector: Optional[str] = None, is_update: bool = False,
                      quantitation_settings: Optional[dict] = None) -> dict:
        """Export results in the configured formats based on trigger type and settings.
        
        Args:
            peaks: List of Peak objects
            d_path: Path to the .D directory
            trigger_type: Type of trigger ('integration', 'ms_search', 'assignment', 'batch')
            detector: Detector name (if None, will get from app.data_handler)
            is_update: True if this is an update to existing results, False for new export
            quantitation_settings: Optional quantitation settings to include in export
            
        Returns:
            dict: Status of exports {'json': bool, 'csv': bool, 'messages': list}
        """
        result = {'json': False, 'csv': False, 'messages': []}
        
        # Get detector name if not provided
        if detector is None and self.app and hasattr(self.app, 'data_handler'):
            detector = getattr(self.app.data_handler, 'current_detector', 'Unknown')
        elif detector is None:
            detector = 'Unknown'
        
        # Export JSON if enabled for this trigger
        if self.should_export_json(trigger_type):
            try:
                if is_update:
                    from logic.json_exporter import update_json_with_ms_search_results
                    success = update_json_with_ms_search_results(peaks, d_path, detector, quantitation_settings)
                else:
                    from logic.json_exporter import export_integration_results_to_json
                    success = export_integration_results_to_json(peaks, d_path, detector, quantitation_settings)
                
                result['json'] = success
                if success:
                    result['messages'].append(f"JSON exported successfully")
                else:
                    result['messages'].append(f"JSON export failed")
            except Exception as e:
                result['messages'].append(f"JSON export error: {str(e)}")
        else:
            result['messages'].append(f"JSON export disabled for {trigger_type}")
        
        # Export CSV if enabled for this trigger
        if self.should_export_csv(trigger_type):
            try:
                if self.app and hasattr(self.app, 'export_results_csv'):
                    csv_filename = os.path.join(d_path, "RESULTS.CSV")
                    success = self.app.export_results_csv(csv_filename)
                    result['csv'] = success
                    if success:
                        result['messages'].append(f"CSV exported to RESULTS.CSV")
                    else:
                        result['messages'].append(f"CSV export failed")
                else:
                    result['messages'].append(f"CSV export not available (no app reference)")
            except Exception as e:
                result['messages'].append(f"CSV export error: {str(e)}")
        else:
            result['messages'].append(f"CSV export disabled for {trigger_type}")
        
        return result
    
    def export_after_integration(self, peaks: List[Any], d_path: str, detector: str = None, 
                                quantitation_settings: dict = None) -> dict:
        """Export results after peak integration."""
        return self.export_results(peaks, d_path, 'integration', detector, is_update=False, 
                                  quantitation_settings=quantitation_settings)
    
    def export_after_ms_search(self, peaks: List[Any], d_path: str, detector: str = None,
                              quantitation_settings: dict = None) -> dict:
        """Export results after MS library search."""
        return self.export_results(peaks, d_path, 'ms_search', detector, is_update=True,
                                  quantitation_settings=quantitation_settings)
    
    def export_after_assignment(self, peaks: List[Any], d_path: str, detector: str = None,
                               quantitation_settings: dict = None) -> dict:
        """Export results after manual peak assignment."""
        return self.export_results(peaks, d_path, 'assignment', detector, is_update=True,
                                  quantitation_settings=quantitation_settings)
    
    def export_during_batch(self, peaks: List[Any], d_path: str, detector: str = None,
                           quantitation_settings: dict = None) -> dict:
        """Export results during batch processing."""
        return self.export_results(peaks, d_path, 'batch', detector, is_update=False,
                                  quantitation_settings=quantitation_settings)
    
    def get_export_summary(self) -> str:
        """Get a summary of current export settings.
        
        Returns:
            str: Human-readable summary of export settings
        """
        json_triggers = []
        csv_triggers = []
        
        triggers = ['integration', 'ms_search', 'assignment', 'batch']
        trigger_names = ['Integration', 'MS Search', 'Manual Assignment', 'Batch Processing']
        
        for trigger, name in zip(triggers, trigger_names):
            if self.should_export_json(trigger):
                json_triggers.append(name)
            if self.should_export_csv(trigger):
                csv_triggers.append(name)
        
        summary = []
        if json_triggers:
            summary.append(f"JSON: {', '.join(json_triggers)}")
        if csv_triggers:
            summary.append(f"CSV: {', '.join(csv_triggers)}")
            
        return "; ".join(summary) if summary else "No automatic exports enabled"
