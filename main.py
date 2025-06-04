import sys
from PySide6.QtWidgets import QApplication, QMessageBox

def main():
    """Main entry point for the application."""
    # Create the application
    app = QApplication(sys.argv)
    
    try:
        # Import here to catch any import errors before showing the window
        from ui.app import ChromaKitApp
        
        # Create and show the main window
        main_window = ChromaKitApp()
        main_window.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except ImportError as e:
        # Handle missing dependencies
        error_message = f"Error importing required modules: {str(e)}\n\n"
        
        if "rainbow" in str(e):
            error_message += (
                "The rainbow-api module is required for GC-MS data processing.\n"
                "You can run in demo mode without data loading capabilities."
            )
            
            # Try to run in demo mode
            try:
                from ui.app import ChromaKitApp
                main_window = ChromaKitApp()
                main_window.show()
                
                # Start the event loop
                sys.exit(app.exec())
            except Exception as inner_e:
                error_message += f"\n\nError starting in demo mode: {str(inner_e)}"
                QMessageBox.critical(None, "Error", error_message)
                sys.exit(1)
        else:
            QMessageBox.critical(None, "Error", error_message)
            sys.exit(1)
    
    except Exception as e:
        # Handle any other startup errors
        error_message = f"Error starting application: {str(e)}"
        QMessageBox.critical(None, "Error", error_message)
        sys.exit(1)

if __name__ == "__main__":
    main()