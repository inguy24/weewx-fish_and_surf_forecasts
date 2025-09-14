#!/usr/bin/env python3
# Magic Animal: Tazmanian Devil
"""
WeeWX Surf & Fishing Forecast Extension Installer
Phase II: Local Surf & Fishing Forecast System

Copyright 2025 Shane Burkhardt
"""

import os
import configobj
import json
import urllib.request
import urllib.parse
import urllib.error
import sys
import subprocess
import time
import yaml
import math
import curses
import textwrap
from typing import Dict, List, Optional, Any, Tuple

# CRITICAL: Correct import path for WeeWX 5.1
try:
    from weecfg.extension import ExtensionInstaller
    import weewx.manager
    import weewx
    import weewx.units
    import weeutil.logger
    log = weeutil.logger.logging.getLogger(__name__)
except ImportError:
    print("Error: This installer requires WeeWX 5.1 or later")
    sys.exit(1)

# CORE ICONS: Consistent with Phase I patterns
CORE_ICONS = {
    'navigation': '📍',    # Location/station selection
    'status': '✅',        # Success indicators  
    'warning': '⚠️',       # Warnings/issues
    'selection': '🔧'      # Configuration/selection
}

# REQUIRED: Loader function for WeeWX extension system
def loader():
    return SurfFishingInstaller()


class InstallationProgressManager:
    """Progress indicator for long operations"""
    
    def __init__(self):
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current_step = 0
        
    def show_step_progress(self, step_name, current=None, total=None):
        """Show progress for a step with optional counter"""
        if current is not None and total is not None:
            char = self.spinner_chars[current % len(self.spinner_chars)]
            print(f"\r  {char} {step_name}... {current}/{total}", end='', flush=True)
        else:
            print(f"  {step_name}...", end='', flush=True)
    
    def complete_step(self, step_name):
        """Mark step as complete"""
        print(f"\r  {step_name}... {CORE_ICONS['status']}")
        
    def show_error(self, step_name, error_msg):
        """Show step error"""
        print(f"\r  {step_name}... {CORE_ICONS['warning']}  {error_msg}")


class SurfFishingPointManager:
    """
    Cursor UI management system for surf and fishing points with real-time weewx.conf persistence
    """
    
    def __init__(self, yaml_data, gebco_client=None):
        """Initialize point manager with YAML configuration and optional GEBCO client"""
        self.yaml_data = yaml_data
        self.gebco_client = gebco_client
        self.config_path = None
        self.current_spots = {'surf_spots': {}, 'fishing_spots': {}}
        self.menu_state = 'main'
        self.selected_index = 0
        
        # Core icons for consistency
        self.CORE_ICONS = {
            'navigation': '📍',
            'status': '✅',
            'warning': '⚠️',
            'selection': '🔧'
        }
    
    def _find_weewx_config_path(self) -> Optional[str]:
        """Find weewx.conf file following WeeWX 5.1 standard locations"""
        possible_paths = [
            '/etc/weewx/weewx.conf',
            '/home/weewx/weewx.conf',
            '~/weewx-data/weewx.conf',
            '/opt/weewx/weewx.conf'
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        return None
    
    def _load_current_spots_from_conf(self) -> bool:
        """Load existing spots from weewx.conf using WeeWX 5.1 thread-safe ConfigObj operations"""
        try:
            if not self.config_path:
                self.config_path = self._find_weewx_config_path()
                if not self.config_path:
                    return False
            
            # WeeWX 5.1 best practice: disable interpolation to prevent ConfigObj format issues
            config = configobj.ConfigObj(self.config_path, interpolation=False)
            
            # Navigate to SurfFishingService section
            service_config = config.get('SurfFishingService', {})
            
            # Load surf spots
            surf_spots = service_config.get('surf_spots', {})
            self.current_spots['surf_spots'] = {str(k): v for k, v in surf_spots.items()}
            
            # Load fishing spots  
            fishing_spots = service_config.get('fishing_spots', {})
            self.current_spots['fishing_spots'] = {str(k): v for k, v in fishing_spots.items()}
            
            return True
            
        except Exception as e:
            # Graceful fallback - start with empty spots if config read fails
            self.current_spots = {'surf_spots': {}, 'fishing_spots': {}}
            return False
    
    def _save_spots_to_conf(self) -> bool:
        """Save current spots to weewx.conf using WeeWX 5.1 thread-safe write operations"""
        try:
            if not self.config_path:
                return False
            
            # WeeWX 5.1 best practice: Read, modify, write pattern with proper ConfigObj handling
            config = configobj.ConfigObj(self.config_path, interpolation=False)
            
            # Ensure SurfFishingService section exists
            if 'SurfFishingService' not in config:
                config['SurfFishingService'] = {}
            
            # Update surf spots section
            config['SurfFishingService']['surf_spots'] = {str(k): v for k, v in self.current_spots['surf_spots'].items()}
            
            # Update fishing spots section
            config['SurfFishingService']['fishing_spots'] = {str(k): v for k, v in self.current_spots['fishing_spots'].items()}
            
            # WeeWX 5.1 best practice: Immediate write to persist changes
            config.write()
            
            return True
            
        except Exception as e:
            return False
    
    def _reset_bathymetry_flag(self, spot_type: str, spot_key: str) -> bool:
        """Reset bathymetry_calculated flag to false for spot recalculation"""
        try:
            if spot_key in self.current_spots[spot_type]:
                self.current_spots[spot_type][spot_key]['bathymetry_calculated'] = 'false'
                return self._save_spots_to_conf()
            return False
        except Exception:
            return False
    
    def _get_next_spot_key(self, spot_type: str) -> str:
        """Generate next available spot key (spot_0, spot_1, etc.)"""
        existing_keys = list(self.current_spots[spot_type].keys())
        
        # Find highest numeric suffix
        max_index = -1
        for key in existing_keys:
            if key.startswith('spot_'):
                try:
                    index = int(key.split('_')[1])
                    max_index = max(max_index, index)
                except (ValueError, IndexError):
                    continue
        
        return f'spot_{max_index + 1}'
    
    def _validate_coordinates(self, lat_str: str, lon_str: str) -> Tuple[Optional[float], Optional[float]]:
        """Validate and convert coordinate strings to float values"""
        try:
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Basic range validation
            if not (-90 <= lat <= 90):
                return None, None
            if not (-180 <= lon <= 180):
                return None, None
                
            return lat, lon
        except ValueError:
            return None, None
    
    def run_management_interface(self) -> Dict[str, Any]:
        """
        Main entry point - run cursor UI management interface
        Returns configuration data for integration with existing install.py workflow
        """
        
        # Load existing configuration
        config_loaded = self._load_current_spots_from_conf()
        
        # Run curses interface if terminal supports it
        try:
            result = curses.wrapper(self._run_curses_interface)
            
            # Convert current_spots format to install.py expected format
            selected_locations = self._convert_to_installer_format()
            
            return {
                'surf_spots': selected_locations['surf_spots'],
                'fishing_spots': selected_locations['fishing_spots'],
                'config_loaded': config_loaded,
                'changes_made': result.get('changes_made', False)
            }
            
        except Exception:
            # Fallback to text interface if curses fails
            return self._run_text_fallback_interface()
    
    def _run_curses_interface(self, stdscr) -> Dict[str, Any]:
        """Main curses interface loop with navigation and spot management"""
        
        curses.curs_set(0)  # Hide cursor
        curses.use_default_colors()
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # Header
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected
        curses.init_pair(3, curses.COLOR_GREEN, -1)                  # Success
        curses.init_pair(4, curses.COLOR_RED, -1)                    # Warning
        curses.init_pair(5, curses.COLOR_YELLOW, -1)                 # Info
        
        changes_made = False
        
        while True:
            stdscr.clear()
            
            if self.menu_state == 'main':
                action = self._display_main_menu(stdscr)
                if action == 'exit':
                    break
                elif action == 'surf_spots':
                    self.menu_state = 'surf_management'
                    self.selected_index = 0
                elif action == 'fishing_spots':
                    self.menu_state = 'fishing_management' 
                    self.selected_index = 0
            
            elif self.menu_state == 'surf_management':
                action = self._display_spot_management(stdscr, 'surf_spots')
                if action == 'back':
                    self.menu_state = 'main'
                    self.selected_index = 0
                elif action.startswith('action_'):
                    changes_made = True
            
            elif self.menu_state == 'fishing_management':
                action = self._display_spot_management(stdscr, 'fishing_spots')
                if action == 'back':
                    self.menu_state = 'main'
                    self.selected_index = 0
                elif action.startswith('action_'):
                    changes_made = True
            
            stdscr.refresh()
        
        return {'changes_made': changes_made}
    
    def _display_main_menu(self, stdscr) -> str:
        """Display main menu with surf/fishing spot management options"""
        
        height, width = stdscr.getmaxyx()
        
        # Header
        header = f"{self.CORE_ICONS['selection']} Surf & Fishing Point Management System"
        stdscr.addstr(0, (width - len(header)) // 2, header, curses.color_pair(1) | curses.A_BOLD)
        
        # Current status
        surf_count = len(self.current_spots['surf_spots'])
        fishing_count = len(self.current_spots['fishing_spots'])
        status = f"Current: {surf_count} surf spots, {fishing_count} fishing spots"
        stdscr.addstr(2, (width - len(status)) // 2, status, curses.color_pair(5))
        
        # Menu options
        menu_items = [
            "Manage Surf Spots",
            "Manage Fishing Spots", 
            "Exit and Continue Installation"
        ]
        
        start_y = height // 2 - len(menu_items) // 2
        
        for i, item in enumerate(menu_items):
            y = start_y + i * 2
            if i == self.selected_index:
                stdscr.addstr(y, (width - len(item)) // 2, f"> {item} <", curses.color_pair(2) | curses.A_BOLD)
            else:
                stdscr.addstr(y, (width - len(item)) // 2, f"  {item}  ", curses.A_NORMAL)
        
        # Instructions
        instructions = "↑↓ Navigate, Enter Select, q Quit"
        stdscr.addstr(height - 2, (width - len(instructions)) // 2, instructions, curses.color_pair(5))
        
        # Handle input
        key = stdscr.getch()
        
        if key == curses.KEY_UP:
            self.selected_index = (self.selected_index - 1) % len(menu_items)
        elif key == curses.KEY_DOWN:
            self.selected_index = (self.selected_index + 1) % len(menu_items)
        elif key == ord('\n') or key == ord(' '):
            if self.selected_index == 0:
                return 'surf_spots'
            elif self.selected_index == 1:
                return 'fishing_spots'
            elif self.selected_index == 2:
                return 'exit'
        elif key == ord('q') or key == 27:  # ESC
            return 'exit'
        
        return 'continue'
    
    def _display_spot_management(self, stdscr, spot_type: str) -> str:
        """Display spot management interface for surf or fishing spots"""
        
        height, width = stdscr.getmaxyx()
        spots = self.current_spots[spot_type]
        spot_list = list(spots.items())
        
        # Header
        type_name = "Surf" if spot_type == 'surf_spots' else "Fishing"
        header = f"{self.CORE_ICONS['navigation']} {type_name} Spot Management ({len(spots)} spots)"
        stdscr.addstr(0, (width - len(header)) // 2, header, curses.color_pair(1) | curses.A_BOLD)
        
        # Spot list with actions
        if spots:
            start_y = 3
            max_display = min(height - 8, len(spot_list))
            
            for i, (spot_key, spot_config) in enumerate(spot_list[:max_display]):
                y = start_y + i
                name = spot_config.get('name', spot_key)
                lat = spot_config.get('latitude', 'N/A')
                lon = spot_config.get('longitude', 'N/A')
                
                # Show validation status for surf spots with detailed information
                if spot_type == 'surf_spots':
                    validation_status = self._get_surf_spot_validation_status(spot_config)
                    bathy_status = f" [{validation_status['icon']} {validation_status['display_text']}]"
                else:
                    bathy_status = ""
                
                spot_text = f"{name} ({lat}, {lon}){bathy_status}"
                
                if i == self.selected_index:
                    stdscr.addstr(y, 2, f"> {spot_text}", curses.color_pair(2) | curses.A_BOLD)
                else:
                    stdscr.addstr(y, 4, spot_text, curses.A_NORMAL)
        else:
            stdscr.addstr(3, (width - 20) // 2, "No spots configured", curses.color_pair(4))
        
        # Action menu
        actions_y = height - 6
        stdscr.addstr(actions_y, 2, "Actions:", curses.A_BOLD)
        stdscr.addstr(actions_y + 1, 4, "a - Add New Spot")
        
        if spots:
            stdscr.addstr(actions_y + 2, 4, "e - Edit Selected Spot")
            stdscr.addstr(actions_y + 3, 4, "d - Delete Selected Spot")
            if spot_type == 'surf_spots':
                stdscr.addstr(actions_y + 4, 4, "r - Reset Bathymetry Flag")
                stdscr.addstr(actions_y + 5, 4, "i - Show Depth Info")
        
        stdscr.addstr(height - 2, 4, "↑↓ Navigate, b Back to Main Menu, q Quit", curses.color_pair(5))
        
        # Handle input
        key = stdscr.getch()
        
        if key == curses.KEY_UP and spots:
            self.selected_index = max(0, self.selected_index - 1)
        elif key == curses.KEY_DOWN and spots:
            self.selected_index = min(len(spot_list) - 1, self.selected_index + 1)
        elif key == ord('a'):
            return self._add_new_spot_dialog(stdscr, spot_type)
        elif key == ord('e') and spots:
            return self._edit_spot_dialog(stdscr, spot_type, spot_list[self.selected_index])
        elif key == ord('d') and spots:
            return self._delete_spot_dialog(stdscr, spot_type, spot_list[self.selected_index])
        elif key == ord('r') and spots and spot_type == 'surf_spots':
            return self._reset_bathymetry_dialog(stdscr, spot_list[self.selected_index])
        elif key == ord('i') and spots and spot_type == 'surf_spots':
            return self._show_validation_info_dialog(stdscr, spot_type, spot_list[self.selected_index])
        elif key == ord('b'):
            return 'back'
        elif key == ord('q') or key == 27:  # ESC
            return 'back'
        
        return 'continue'
    
    def _add_new_spot_dialog(self, stdscr, spot_type: str) -> str:
        """Enhanced dialog for adding a new surf or fishing spot with all characteristics"""
        
        height, width = stdscr.getmaxyx()
        
        # Create larger dialog window
        dialog_height = 18 if spot_type == 'surf_spots' else 16
        dialog_width = 70
        dialog_y = (height - dialog_height) // 2
        dialog_x = (width - dialog_width) // 2
        
        dialog_win = curses.newwin(dialog_height, dialog_width, dialog_y, dialog_x)
        dialog_win.box()
        
        type_name = "Surf" if spot_type == 'surf_spots' else "Fishing"
        dialog_win.addstr(1, 2, f"Add New {type_name} Spot", curses.A_BOLD)
        
        # Input fields with defaults
        if spot_type == 'surf_spots':
            fields = ['Name', 'Latitude', 'Longitude', 'Beach Angle (0-360)', 'Bottom Type', 'Exposure']
            field_values = ['', '', '', '270', 'sand', 'exposed']
        else:
            fields = ['Name', 'Latitude', 'Longitude', 'Location Type', 'Target Category']
            field_values = ['', '', '', 'shore', 'mixed_bag']
        
        current_field = 0
        curses.curs_set(1)  # Show cursor for input
        
        # Same options dictionaries as in edit dialog
        bottom_type_options = {'1': 'sand', '2': 'reef', '3': 'point', '4': 'jetty', '5': 'mixed'}
        bottom_type_display = {'sand': '1-Sand', 'reef': '2-Reef', 'point': '3-Point', 'jetty': '4-Jetty', 'mixed': '5-Mixed'}
        exposure_options = {'1': 'exposed', '2': 'semi_protected', '3': 'protected'}
        exposure_display = {'exposed': '1-Exposed', 'semi_protected': '2-Semi-protected', 'protected': '3-Protected'}
        location_type_options = {'1': 'shore', '2': 'pier', '3': 'boat', '4': 'mixed'}
        location_type_display = {'shore': '1-Shore', 'pier': '2-Pier', 'boat': '3-Boat', 'mixed': '4-Mixed'}
        
        # Get fish categories for target category field
        target_category_options = {}
        target_category_display = {}
        if hasattr(self, 'yaml_data') and self.yaml_data and 'fish_categories' in self.yaml_data:
            fish_categories = self.yaml_data['fish_categories']
            for i, (cat_key, cat_data) in enumerate(fish_categories.items(), 1):
                target_category_options[str(i)] = cat_key
                display_name = cat_data.get('display_name', cat_key.replace('_', ' ').title())
                target_category_display[cat_key] = f"{i}-{display_name}"
        
        while True:
            # Clear and redraw dialog (same logic as edit dialog)
            for y in range(2, dialog_height - 1):
                dialog_win.addstr(y, 1, " " * (dialog_width - 2))
            
            # Display fields (same logic as edit dialog)
            for i, field in enumerate(fields):
                y = 3 + i * 2
                
                dialog_win.addstr(y, 2, f"{field}:", curses.A_BOLD if i == current_field else curses.A_NORMAL)
                
                # Handle display values for dropdown fields
                field_display_value = field_values[i][:40]
                if spot_type == 'surf_spots':
                    if 'Bottom Type' in field:
                        field_display_value = bottom_type_display.get(field_values[i], field_values[i])
                    elif 'Exposure' in field:
                        field_display_value = exposure_display.get(field_values[i], field_values[i])
                else:
                    if 'Location Type' in field:
                        field_display_value = location_type_display.get(field_values[i], field_values[i])
                    elif 'Target Category' in field:
                        field_display_value = target_category_display.get(field_values[i], field_values[i])
                
                dialog_win.addstr(y, 20, " " * 45)
                dialog_win.addstr(y, 20, field_display_value)
                
                if i == current_field:
                    cursor_pos = min(len(field_display_value), 40)
                    if 'Type' in field or 'Exposure' in field or 'Category' in field:
                        dialog_win.addstr(y, 20 + cursor_pos, " <-", curses.A_REVERSE)
                    else:
                        dialog_win.addch(y, 20 + cursor_pos, curses.ACS_BLOCK)
            
            # Instructions
            instructions_y = dialog_height - 4
            dialog_win.addstr(instructions_y, 2, " " * (dialog_width - 4))
            current_field_name = fields[current_field]
            if 'Type' in current_field_name or 'Exposure' in current_field_name or 'Category' in current_field_name:
                dialog_win.addstr(instructions_y, 2, "1-5: Select option, Tab: Next field", curses.color_pair(5))
            else:
                dialog_win.addstr(instructions_y, 2, "Type to edit, Tab: Next field", curses.color_pair(5))
            
            dialog_win.addstr(dialog_height - 3, 2, "Enter: Save, Esc: Cancel")
            dialog_win.refresh()
            
            key = dialog_win.getch()
            
            if key == 27:  # ESC - Cancel
                curses.curs_set(0)
                return 'continue'
            elif key == ord('\t'):  # Tab - Next field
                current_field = (current_field + 1) % len(fields)
            elif key == ord('\n'):  # Enter - Save
                # Validate and save
                if self._validate_spot_input(field_values, spot_type):
                    self._save_new_spot(field_values, spot_type)
                    curses.curs_set(0)
                    return 'action_added'
                else:
                    dialog_win.addstr(dialog_height - 2, 2, "Invalid input - check coordinates!", curses.color_pair(4))
                    dialog_win.refresh()
                    curses.napms(2000)
            
            # Handle field-specific input (same logic as edit dialog)
            elif 'Bottom Type' in current_field_name and spot_type == 'surf_spots':
                if chr(key) in bottom_type_options:
                    field_values[current_field] = bottom_type_options[chr(key)]
            elif 'Exposure' in current_field_name and spot_type == 'surf_spots':
                if chr(key) in exposure_options:
                    field_values[current_field] = exposure_options[chr(key)]
            elif 'Location Type' in current_field_name and spot_type == 'fishing_spots':
                if chr(key) in location_type_options:
                    field_values[current_field] = location_type_options[chr(key)]
            elif 'Target Category' in current_field_name and spot_type == 'fishing_spots':
                if chr(key) in target_category_options:
                    field_values[current_field] = target_category_options[chr(key)]
            
            # Handle text input for non-dropdown fields
            elif not any(x in current_field_name for x in ['Type', 'Exposure', 'Category']):
                if key == curses.KEY_BACKSPACE or key == 127:
                    if field_values[current_field]:
                        field_values[current_field] = field_values[current_field][:-1]
                elif 32 <= key <= 126:  # Printable characters
                    if len(field_values[current_field]) < 30:
                        field_values[current_field] += chr(key)
    
    def _edit_spot_dialog(self, stdscr, spot_type: str, spot_item: Tuple[str, Dict]) -> str:
        """Enhanced dialog for editing existing spot with full surf characteristics"""
        
        spot_key, spot_config = spot_item
        
        height, width = stdscr.getmaxyx()
        
        # Create larger dialog window to accommodate all fields
        dialog_height = 18 if spot_type == 'surf_spots' else 16
        dialog_width = 70
        dialog_y = (height - dialog_height) // 2
        dialog_x = (width - dialog_width) // 2
        
        dialog_win = curses.newwin(dialog_height, dialog_width, dialog_y, dialog_x)
        dialog_win.box()
        
        type_name = "Surf" if spot_type == 'surf_spots' else "Fishing"
        dialog_win.addstr(1, 2, f"Edit {type_name} Spot: {spot_config.get('name', spot_key)}", curses.A_BOLD)
        
        # Pre-populate with current values
        field_values = [
            spot_config.get('name', ''),
            spot_config.get('latitude', ''),
            spot_config.get('longitude', '')
        ]
        
        if spot_type == 'surf_spots':
            field_values.extend([
                spot_config.get('beach_facing', ''),
                spot_config.get('bottom_type', 'sand'),
                spot_config.get('exposure', 'exposed')
            ])
            fields = ['Name', 'Latitude', 'Longitude', 'Beach Angle', 'Bottom Type', 'Exposure']
        else:  # fishing_spots
            field_values.extend([
                spot_config.get('location_type', 'shore'),
                spot_config.get('target_category', 'mixed_bag')
            ])
            fields = ['Name', 'Latitude', 'Longitude', 'Location Type', 'Target Category']
        
        current_field = 0
        curses.curs_set(1)
        
        # Options for dropdown fields
        bottom_type_options = {
            '1': 'sand', '2': 'reef', '3': 'point', '4': 'jetty', '5': 'mixed'
        }
        bottom_type_display = {
            'sand': '1-Sand', 'reef': '2-Reef', 'point': '3-Point', 'jetty': '4-Jetty', 'mixed': '5-Mixed'
        }
        
        exposure_options = {
            '1': 'exposed', '2': 'semi_protected', '3': 'protected'
        }
        exposure_display = {
            'exposed': '1-Exposed', 'semi_protected': '2-Semi-protected', 'protected': '3-Protected'
        }
        
        location_type_options = {
            '1': 'shore', '2': 'pier', '3': 'boat', '4': 'mixed'
        }
        location_type_display = {
            'shore': '1-Shore', 'pier': '2-Pier', 'boat': '3-Boat', 'mixed': '4-Mixed'
        }
        
        # Get fish categories for target category field (if available)
        target_category_options = {}
        target_category_display = {}
        if hasattr(self, 'yaml_data') and self.yaml_data and 'fish_categories' in self.yaml_data:
            fish_categories = self.yaml_data['fish_categories']
            for i, (cat_key, cat_data) in enumerate(fish_categories.items(), 1):
                target_category_options[str(i)] = cat_key
                # Truncate long display names to prevent overflow
                display_name = cat_data.get('display_name', cat_key.replace('_', ' ').title())
                display_name = display_name[:20]  # Limit display name length
                target_category_display[cat_key] = f"{i}-{display_name}"
        
        while True:
            # Clear and redraw dialog content
            dialog_win.clear()
            dialog_win.box()
            dialog_win.addstr(1, 2, f"Edit {type_name} Spot: {spot_config.get('name', spot_key)}", curses.A_BOLD)
            
            # Display fields with enhanced formatting
            for i, field in enumerate(fields):
                y = 3 + i * 2
                
                dialog_win.addstr(y, 2, f"{field}:", curses.A_BOLD if i == current_field else curses.A_NORMAL)
                
                # Handle display values for dropdown fields
                field_display_value = field_values[i][:40]
                if spot_type == 'surf_spots':
                    if 'Bottom Type' in field:
                        field_display_value = bottom_type_display.get(field_values[i], field_values[i])
                    elif 'Exposure' in field:
                        field_display_value = exposure_display.get(field_values[i], field_values[i])
                else:
                    if 'Location Type' in field:
                        field_display_value = location_type_display.get(field_values[i], field_values[i])
                    elif 'Target Category' in field:
                        field_display_value = target_category_display.get(field_values[i], field_values[i])
                
                dialog_win.addstr(y, 20, " " * 45)
                dialog_win.addstr(y, 20, field_display_value)
                
                if i == current_field:
                    cursor_pos = min(len(field_display_value), 40)
                    if 'Type' in field or 'Exposure' in field or 'Category' in field:
                        dialog_win.addstr(y, 20 + cursor_pos, " <-", curses.A_REVERSE)
                    else:
                        dialog_win.addch(y, 20 + cursor_pos, curses.ACS_BLOCK)
            
            # Instructions based on current field type
            instructions_y = dialog_height - 4
            for clear_y in range(instructions_y, dialog_height - 1):
                dialog_win.addstr(clear_y, 1, " " * (dialog_width - 2))
            
            current_field_name = fields[current_field]
            if current_field_name in ['Bottom Type', 'Exposure', 'Location Type', 'Target Category']:
                dialog_win.addstr(instructions_y, 2, "1-5: Select option, Tab: Next field", curses.color_pair(5))
            else:
                dialog_win.addstr(instructions_y, 2, "Type to edit, Tab: Next field", curses.color_pair(5))
            
            dialog_win.addstr(dialog_height - 3, 2, "Enter: Save, Esc: Cancel", curses.color_pair(5))
            dialog_win.refresh()
            
            key = dialog_win.getch()
            
            if key == 27:  # ESC - Cancel
                curses.curs_set(0)
                return 'continue'
            elif key == ord('\t'):  # Tab - Next field
                current_field = (current_field + 1) % len(fields)
            elif key == ord('\n'):  # Enter - Save
                if self._validate_spot_input(field_values, spot_type):
                    # Update spot with enhanced field handling
                    if spot_type == 'surf_spots':
                        enhanced_field_values = field_values[:6]  # All 6 fields
                    else:
                        enhanced_field_values = field_values[:5]  # All 5 fields
                    
                    self._update_existing_spot_enhanced(spot_key, enhanced_field_values, spot_type)
                    curses.curs_set(0)
                    return 'action_updated'
                else:
                    dialog_win.addstr(dialog_height - 2, 2, "Invalid input - check coordinates!", curses.color_pair(4))
                    dialog_win.refresh()
                    curses.napms(2000)
            
            # Handle input based on field type
            elif current_field_name == 'Bottom Type' and spot_type == 'surf_spots':
                if chr(key) in bottom_type_options:
                    field_values[current_field] = bottom_type_options[chr(key)]
            elif current_field_name == 'Exposure' and spot_type == 'surf_spots':
                if chr(key) in exposure_options:
                    field_values[current_field] = exposure_options[chr(key)]
            elif current_field_name == 'Location Type' and spot_type == 'fishing_spots':
                if chr(key) in location_type_options:
                    field_values[current_field] = location_type_options[chr(key)]
            elif current_field_name == 'Target Category' and spot_type == 'fishing_spots':
                if chr(key) in target_category_options:
                    field_values[current_field] = target_category_options[chr(key)]
            
            # Handle text input for non-dropdown fields
            elif current_field_name not in ['Bottom Type', 'Exposure', 'Location Type', 'Target Category']:
                if key == curses.KEY_BACKSPACE or key == 127:
                    if field_values[current_field]:
                        field_values[current_field] = field_values[current_field][:-1]
                elif 32 <= key <= 126:  # Printable characters
                    if len(field_values[current_field]) < 30:
                        field_values[current_field] += chr(key)
    
    def _delete_spot_dialog(self, stdscr, spot_type: str, spot_item: Tuple[str, Dict]) -> str:
        """Confirmation dialog for deleting spot"""
        
        spot_key, spot_config = spot_item
        
        height, width = stdscr.getmaxyx()
        
        # Create confirmation dialog
        dialog_height = 8
        dialog_width = 50
        dialog_y = (height - dialog_height) // 2
        dialog_x = (width - dialog_width) // 2
        
        dialog_win = curses.newwin(dialog_height, dialog_width, dialog_y, dialog_x)
        dialog_win.box()
        
        name = spot_config.get('name', spot_key)
        dialog_win.addstr(1, 2, f"Delete Spot: {name}", curses.A_BOLD)
        dialog_win.addstr(3, 2, "Are you sure? This cannot be undone.")
        dialog_win.addstr(5, 2, "y - Yes, Delete    n - No, Cancel", curses.color_pair(4))
        
        dialog_win.refresh()
        
        while True:
            key = dialog_win.getch()
            if key == ord('y') or key == ord('Y'):
                # Delete spot
                del self.current_spots[spot_type][spot_key]
                self._save_spots_to_conf()
                # Adjust selected_index if needed
                if self.selected_index >= len(self.current_spots[spot_type]):
                    self.selected_index = max(0, len(self.current_spots[spot_type]) - 1)
                return 'action_deleted'
            elif key == ord('n') or key == ord('N') or key == 27:
                return 'continue'
    
    def _reset_bathymetry_dialog(self, stdscr, spot_item: Tuple[str, Dict]) -> str:
        """Dialog for resetting bathymetry calculation flag"""
        
        spot_key, spot_config = spot_item
        
        height, width = stdscr.getmaxyx()
        
        dialog_height = 8
        dialog_width = 55
        dialog_y = (height - dialog_height) // 2
        dialog_x = (width - dialog_width) // 2
        
        dialog_win = curses.newwin(dialog_height, dialog_width, dialog_y, dialog_x)
        dialog_win.box()
        
        name = spot_config.get('name', spot_key)
        dialog_win.addstr(1, 2, f"Reset Bathymetry: {name}", curses.A_BOLD)
        dialog_win.addstr(3, 2, "Reset bathymetry flag to recalculate depths?")
        dialog_win.addstr(5, 2, "y - Yes, Reset    n - Cancel", curses.color_pair(5))
        
        dialog_win.refresh()
        
        while True:
            key = dialog_win.getch()
            if key == ord('y') or key == ord('Y'):
                if self._reset_bathymetry_flag('surf_spots', spot_key):
                    return 'action_reset'
                else:
                    return 'continue'
            elif key == ord('n') or key == ord('N') or key == 27:
                return 'continue'
    
    def _validate_spot_input(self, field_values: List[str], spot_type: str) -> bool:
        """Validate user input for spot creation/editing"""
        
        # Check required fields
        if not field_values[0].strip():  # Name
            return False
        
        # Validate coordinates
        lat, lon = self._validate_coordinates(field_values[1], field_values[2])
        if lat is None or lon is None:
            return False
        
        # Validate beach angle for surf spots
        if spot_type == 'surf_spots' and len(field_values) > 3:
            try:
                angle = float(field_values[3])
                if not (0 <= angle <= 360):
                    return False
            except ValueError:
                return False
        
        return True
    
    def _save_new_spot(self, field_values: List[str], spot_type: str):
        """Save new spot to current configuration"""
        
        spot_key = self._get_next_spot_key(spot_type)
        lat, lon = self._validate_coordinates(field_values[1], field_values[2])
        
        # Create spot configuration using data-driven approach
        spot_config = {
            'name': field_values[0].strip(),
            'latitude': str(lat),
            'longitude': str(lon),
            'type': spot_type.replace('_spots', ''),
            'active': 'true'
        }
        
        if spot_type == 'surf_spots':
            spot_config.update({
                'beach_facing': field_values[3] if len(field_values) > 3 else '270',
                'bottom_type': 'sand',
                'exposure': 'exposed',
                'bathymetry_calculated': 'false'
            })
        else:  # fishing_spots
            spot_config.update({
                'location_type': 'shore',
                'target_category': 'mixed_bag'
            })
        
        # Add to current spots and save to conf
        self.current_spots[spot_type][spot_key] = spot_config
        self._save_spots_to_conf()
    
    def _update_existing_spot(self, spot_key: str, field_values: List[str], spot_type: str):
        """Enhanced update method to handle all surf and fishing characteristics"""
        
        lat, lon = self._validate_coordinates(field_values[1], field_values[2])
        
        # Get existing config and update with new values
        spot_config = self.current_spots[spot_type][spot_key]
        spot_config['name'] = field_values[0].strip()
        spot_config['latitude'] = str(lat)
        spot_config['longitude'] = str(lon)
        
        if spot_type == 'surf_spots':
            # Update surf-specific fields
            if len(field_values) > 3:
                spot_config['beach_facing'] = field_values[3]
            if len(field_values) > 4:
                spot_config['bottom_type'] = field_values[4]
            if len(field_values) > 5:
                spot_config['exposure'] = field_values[5]
            # Reset bathymetry flag when surf characteristics change
            spot_config['bathymetry_calculated'] = 'false'
        else:  # fishing_spots
            # Update fishing-specific fields
            if len(field_values) > 3:
                spot_config['location_type'] = field_values[3]
            if len(field_values) > 4:
                spot_config['target_category'] = field_values[4]
        
        self._save_spots_to_conf()

    def _display_spot_details(self, spot_key: str, spot_config: Dict, spot_type: str):
        """Enhanced display method to show all spot characteristics"""
        
        name = spot_config.get('name', spot_key)
        lat = spot_config.get('latitude', 'Unknown')
        lon = spot_config.get('longitude', 'Unknown')
        
        print(f"  • {name}")
        print(f"    Location: {lat}, {lon}")
        
        if spot_type == 'surf_spots':
            beach_angle = spot_config.get('beach_facing', 'Unknown')
            bottom_type = spot_config.get('bottom_type', 'Unknown')
            exposure = spot_config.get('exposure', 'Unknown')
            bathymetry_calculated = spot_config.get('bathymetry_calculated', 'false')
            
            print(f"    Beach facing: {beach_angle}°")
            print(f"    Bottom type: {bottom_type}")
            print(f"    Exposure: {exposure}")
            print(f"    Bathymetry calculated: {bathymetry_calculated}")
        else:  # fishing_spots
            location_type = spot_config.get('location_type', 'Unknown')
            target_category = spot_config.get('target_category', 'Unknown')
            
            print(f"    Location type: {location_type}")
            print(f"    Target category: {target_category}")
    
    def _run_text_fallback_interface(self) -> Dict[str, Any]:
        """
        Text-based fallback interface when curses is not available
        Provides basic CRUD operations through command-line prompts
        """
        
        print(f"\n{self.CORE_ICONS['selection']} Surf & Fishing Point Management System")
        print("=" * 60)
        
        config_loaded = self._load_current_spots_from_conf()
        changes_made = False
        
        if not config_loaded:
            print(f"{self.CORE_ICONS['warning']} Could not load existing configuration - starting fresh")
        
        while True:
            self._display_text_main_menu()
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == '1':
                changes_made |= self._text_manage_spots('surf_spots')
            elif choice == '2':
                changes_made |= self._text_manage_spots('fishing_spots')
            elif choice == '3':
                self._display_current_configuration()
            elif choice == '4':
                break
            else:
                print(f"{self.CORE_ICONS['warning']} Invalid choice. Please enter 1-4.")
        
        # Convert to installer format
        selected_locations = self._convert_to_installer_format()
        
        return {
            'surf_spots': selected_locations['surf_spots'],
            'fishing_spots': selected_locations['fishing_spots'],
            'config_loaded': config_loaded,
            'changes_made': changes_made
        }
    
    def _display_text_main_menu(self):
        """Display main menu for text interface"""
        
        surf_count = len(self.current_spots['surf_spots'])
        fishing_count = len(self.current_spots['fishing_spots'])
        
        print(f"\nCurrent Configuration: {surf_count} surf spots, {fishing_count} fishing spots")
        print("\nManagement Options:")
        print("1. Manage Surf Spots")
        print("2. Manage Fishing Spots")
        print("3. Display Current Configuration")
        print("4. Exit and Continue Installation")
    
    def _text_manage_spots(self, spot_type: str) -> bool:
        """Text interface for managing specific spot type"""
        
        type_name = "Surf" if spot_type == 'surf_spots' else "Fishing"
        spots = self.current_spots[spot_type]
        changes_made = False
        
        while True:
            print(f"\n{self.CORE_ICONS['navigation']} {type_name} Spot Management")
            print("-" * 40)
            
            if spots:
                print("Current Spots:")
                for i, (spot_key, spot_config) in enumerate(spots.items(), 1):
                    name = spot_config.get('name', spot_key)
                    lat = spot_config.get('latitude', 'N/A')
                    lon = spot_config.get('longitude', 'N/A')
                    
                    if spot_type == 'surf_spots':
                        bathy_calc = spot_config.get('bathymetry_calculated', 'false')
                        bathy_status = f" [Bathymetry: {'Calculated' if bathy_calc == 'true' else 'Needs Calculation'}]"
                    else:
                        bathy_status = ""
                    
                    print(f"  {i}. {name} ({lat}, {lon}){bathy_status}")
            else:
                print("No spots configured.")
            
            print(f"\nActions:")
            print("1. Add New Spot")
            if spots:
                print("2. Edit Spot")
                print("3. Delete Spot")
                if spot_type == 'surf_spots':
                    print("4. Reset Bathymetry Flag")
            print("0. Back to Main Menu")
            
            choice = input(f"\nSelect action: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                changes_made |= self._text_add_spot(spot_type)
            elif choice == '2' and spots:
                changes_made |= self._text_edit_spot(spot_type, spots)
            elif choice == '3' and spots:
                changes_made |= self._text_delete_spot(spot_type, spots)
            elif choice == '4' and spots and spot_type == 'surf_spots':
                changes_made |= self._text_reset_bathymetry(spots)
            else:
                print(f"{self.CORE_ICONS['warning']} Invalid choice.")
        
        return changes_made
    
    def _text_add_spot(self, spot_type: str) -> bool:
        """Text interface for adding new spot"""
        
        type_name = "Surf" if spot_type == 'surf_spots' else "Fishing"
        print(f"\n{self.CORE_ICONS['selection']} Add New {type_name} Spot")
        
        name = input("Spot name: ").strip()
        if not name:
            print(f"{self.CORE_ICONS['warning']} Name cannot be empty.")
            return False
        
        lat_str = input("Latitude (-90 to 90): ").strip()
        lon_str = input("Longitude (-180 to 180): ").strip()
        
        lat, lon = self._validate_coordinates(lat_str, lon_str)
        if lat is None or lon is None:
            print(f"{self.CORE_ICONS['warning']} Invalid coordinates.")
            return False
        
        beach_angle = '270'  # Default
        if spot_type == 'surf_spots':
            beach_str = input("Beach facing angle (0-360, default 270): ").strip()
            if beach_str:
                try:
                    beach_angle = str(float(beach_str))
                    if not (0 <= float(beach_angle) <= 360):
                        print(f"{self.CORE_ICONS['warning']} Beach angle must be 0-360.")
                        return False
                except ValueError:
                    print(f"{self.CORE_ICONS['warning']} Invalid beach angle.")
                    return False
        
        # Create spot using existing logic
        field_values = [name, lat_str, lon_str]
        if spot_type == 'surf_spots':
            field_values.append(beach_angle)
        
        self._save_new_spot(field_values, spot_type)
        print(f"{self.CORE_ICONS['status']} Added {name} successfully.")
        return True
    
    def _text_edit_spot(self, spot_type: str, spots: Dict) -> bool:
        """Text interface for editing existing spot with full surf characteristics"""
        
        spot_list = list(spots.items())
        
        print("Select spot to edit:")
        for i, (spot_key, spot_config) in enumerate(spot_list, 1):
            name = spot_config.get('name', spot_key)
            print(f"  {i}. {name}")
        
        try:
            choice = int(input("Spot number: ").strip()) - 1
            if 0 <= choice < len(spot_list):
                spot_key, spot_config = spot_list[choice]
                
                print(f"\nEditing: {spot_config.get('name', spot_key)}")
                print("(Press Enter to keep current value)")
                
                # Get new values - Name
                name = input(f"Name [{spot_config.get('name', '')}]: ").strip()
                if not name:
                    name = spot_config.get('name', '')
                
                # Get new values - Coordinates
                lat_str = input(f"Latitude [{spot_config.get('latitude', '')}]: ").strip()
                if not lat_str:
                    lat_str = spot_config.get('latitude', '')
                
                lon_str = input(f"Longitude [{spot_config.get('longitude', '')}]: ").strip()
                if not lon_str:
                    lon_str = spot_config.get('longitude', '')
                
                # Validate coordinates
                lat, lon = self._validate_coordinates(lat_str, lon_str)
                if lat is None or lon is None:
                    print(f"{self.CORE_ICONS['warning']} Invalid coordinates.")
                    return False
                
                # Surf-specific characteristics
                if spot_type == 'surf_spots':
                    # Beach angle
                    beach_angle = spot_config.get('beach_facing', '270')
                    beach_str = input(f"Beach angle [{beach_angle}]: ").strip()
                    if beach_str:
                        try:
                            beach_angle = str(float(beach_str))
                            if not (0 <= float(beach_angle) <= 360):
                                print(f"{self.CORE_ICONS['warning']} Beach angle must be 0-360, keeping current.")
                                beach_angle = spot_config.get('beach_facing', '270')
                        except ValueError:
                            print(f"{self.CORE_ICONS['warning']} Invalid beach angle, keeping current.")
                            beach_angle = spot_config.get('beach_facing', '270')
                    
                    # Bottom type selection
                    current_bottom = spot_config.get('bottom_type', 'sand')
                    print(f"\nBottom type (currently: {current_bottom}):")
                    print("  1. Sand (Beach break)")
                    print("  2. Reef (Coral/rock reef)")
                    print("  3. Point break (Rocky point)")
                    print("  4. Jetty/Pier (Man-made structure)")
                    print("  5. Mixed")
                    print("  [Enter] Keep current")
                    
                    bottom_choice = input("Select bottom type (1-5): ").strip()
                    if bottom_choice:
                        bottom_type = {
                            "1": "sand", 
                            "2": "reef", 
                            "3": "point", 
                            "4": "jetty", 
                            "5": "mixed"
                        }.get(bottom_choice)
                        if bottom_type:
                            current_bottom = bottom_type
                        else:
                            print(f"{self.CORE_ICONS['warning']} Invalid choice, keeping current bottom type.")
                    
                    # Exposure selection
                    current_exposure = spot_config.get('exposure', 'exposed')
                    print(f"\nExposure to ocean swell (currently: {current_exposure}):")
                    print("  1. Fully exposed (open ocean)")
                    print("  2. Semi-protected (bay/cove)")
                    print("  3. Protected (harbor/inlet)")
                    print("  [Enter] Keep current")
                    
                    exposure_choice = input("Select exposure (1-3): ").strip()
                    if exposure_choice:
                        exposure_type = {
                            "1": "exposed", 
                            "2": "semi_protected", 
                            "3": "protected"
                        }.get(exposure_choice)
                        if exposure_type:
                            current_exposure = exposure_type
                        else:
                            print(f"{self.CORE_ICONS['warning']} Invalid choice, keeping current exposure.")
                    
                    # Update spot with all surf characteristics
                    field_values = [name, lat_str, lon_str, beach_angle, current_bottom, current_exposure]
                    
                else:  # fishing_spots
                    # For fishing spots, get target category if available
                    current_target = spot_config.get('target_category', 'mixed_bag')
                    current_location = spot_config.get('location_type', 'shore')
                    
                    # Location type selection
                    print(f"\nFishing location type (currently: {current_location}):")
                    print("  1. Shore/Beach (fishing from land)")
                    print("  2. Pier/Jetty (fishing from structure)")
                    print("  3. Boat (fishing from vessel)")
                    print("  4. Mixed (multiple methods)")
                    print("  [Enter] Keep current")
                    
                    location_choice = input("Select location type (1-4): ").strip()
                    if location_choice:
                        location_type = {
                            "1": "shore", 
                            "2": "pier", 
                            "3": "boat", 
                            "4": "mixed"
                        }.get(location_choice)
                        if location_type:
                            current_location = location_type
                        else:
                            print(f"{self.CORE_ICONS['warning']} Invalid choice, keeping current location type.")
                    
                    # Target species category selection (if fish_categories available)
                    if hasattr(self, 'yaml_data') and self.yaml_data and 'fish_categories' in self.yaml_data:
                        fish_categories = self.yaml_data['fish_categories']
                        print(f"\nPrimary target fish category (currently: {current_target}):")
                        category_keys = list(fish_categories.keys())
                        for i, (category_key, category_data) in enumerate(fish_categories.items(), 1):
                            display_name = category_data.get('display_name', category_key.replace('_', ' ').title())
                            species_list = category_data.get('species', [])
                            if isinstance(species_list, list):
                                species_str = ', '.join(species_list[:3]) + ('...' if len(species_list) > 3 else '')
                            else:
                                species_str = str(species_list)
                            print(f"    {i}. {display_name} ({species_str})")
                        print("  [Enter] Keep current")
                        
                        category_choice = input("Select fish category: ").strip()
                        if category_choice:
                            try:
                                choice_idx = int(category_choice) - 1
                                if 0 <= choice_idx < len(category_keys):
                                    current_target = category_keys[choice_idx]
                                else:
                                    print(f"{self.CORE_ICONS['warning']} Invalid choice, keeping current target category.")
                            except ValueError:
                                print(f"{self.CORE_ICONS['warning']} Invalid input, keeping current target category.")
                    
                    # Update spot with all fishing characteristics
                    field_values = [name, lat_str, lon_str, current_location, current_target]
                
                # Update the spot using enhanced method
                self._update_existing_spot_enhanced(spot_key, field_values, spot_type)
                print(f"{self.CORE_ICONS['status']} Updated {name} successfully.")
                return True
            else:
                print(f"{self.CORE_ICONS['warning']} Invalid spot number.")
                return False
        except ValueError:
            print(f"{self.CORE_ICONS['warning']} Invalid input.")
            return False
    
    def _text_delete_spot(self, spot_type: str, spots: Dict) -> bool:
        """Text interface for deleting spot"""
        
        spot_list = list(spots.items())
        
        print("Select spot to delete:")
        for i, (spot_key, spot_config) in enumerate(spot_list, 1):
            name = spot_config.get('name', spot_key)
            print(f"  {i}. {name}")
        
        try:
            choice = int(input("Spot number: ").strip()) - 1
            if 0 <= choice < len(spot_list):
                spot_key, spot_config = spot_list[choice]
                name = spot_config.get('name', spot_key)
                
                confirm = input(f"Delete '{name}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    del self.current_spots[spot_type][spot_key]
                    self._save_spots_to_conf()
                    print(f"{self.CORE_ICONS['status']} Deleted {name} successfully.")
                    return True
                else:
                    print("Cancelled.")
                    return False
            else:
                print(f"{self.CORE_ICONS['warning']} Invalid spot number.")
                return False
        except ValueError:
            print(f"{self.CORE_ICONS['warning']} Invalid input.")
            return False
    
    def _text_reset_bathymetry(self, spots: Dict) -> bool:
        """Text interface for resetting bathymetry flag"""
        
        spot_list = list(spots.items())
        
        print("Select surf spot to reset bathymetry:")
        for i, (spot_key, spot_config) in enumerate(spot_list, 1):
            name = spot_config.get('name', spot_key)
            bathy_calc = spot_config.get('bathymetry_calculated', 'false')
            status = 'Calculated' if bathy_calc == 'true' else 'Needs Calculation'
            print(f"  {i}. {name} [{status}]")
        
        try:
            choice = int(input("Spot number: ").strip()) - 1
            if 0 <= choice < len(spot_list):
                spot_key, spot_config = spot_list[choice]
                name = spot_config.get('name', spot_key)
                
                confirm = input(f"Reset bathymetry flag for '{name}'? (y/n): ").strip().lower()
                if confirm == 'y':
                    if self._reset_bathymetry_flag('surf_spots', spot_key):
                        print(f"{self.CORE_ICONS['status']} Reset bathymetry flag for {name}.")
                        return True
                    else:
                        print(f"{self.CORE_ICONS['warning']} Failed to reset flag.")
                        return False
                else:
                    print("Cancelled.")
                    return False
            else:
                print(f"{self.CORE_ICONS['warning']} Invalid spot number.")
                return False
        except ValueError:
            print(f"{self.CORE_ICONS['warning']} Invalid input.")
            return False
    
    def _display_current_configuration(self):
        """Display complete current configuration"""
        
        print(f"\n{self.CORE_ICONS['navigation']} Current Configuration")
        print("=" * 50)
        
        # Display surf spots
        surf_spots = self.current_spots['surf_spots']
        print(f"\nSurf Spots ({len(surf_spots)}):")
        if surf_spots:
            for spot_key, spot_config in surf_spots.items():
                name = spot_config.get('name', spot_key)
                lat = spot_config.get('latitude', 'N/A')
                lon = spot_config.get('longitude', 'N/A')
                beach_angle = spot_config.get('beach_facing', 'N/A')
                bathy_calc = spot_config.get('bathymetry_calculated', 'false')
                
                print(f"  • {name}")
                print(f"    Location: {lat}, {lon}")
                print(f"    Beach Facing: {beach_angle}°")
                print(f"    Bathymetry: {'Calculated' if bathy_calc == 'true' else 'Needs Calculation'}")
        else:
            print("  None configured")
        
        # Display fishing spots
        fishing_spots = self.current_spots['fishing_spots']
        print(f"\nFishing Spots ({len(fishing_spots)}):")
        if fishing_spots:
            for spot_key, spot_config in fishing_spots.items():
                name = spot_config.get('name', spot_key)
                lat = spot_config.get('latitude', 'N/A')
                lon = spot_config.get('longitude', 'N/A')
                location_type = spot_config.get('location_type', 'shore')
                
                print(f"  • {name}")
                print(f"    Location: {lat}, {lon}")
                print(f"    Type: {location_type}")
        else:
            print("  None configured")
    
    def _convert_to_installer_format(self) -> Dict[str, List[Dict]]:
        """Convert current spots to format expected by install.py workflow"""
        
        result = {
            'surf_spots': [],
            'fishing_spots': []
        }
        
        # Convert surf spots
        for spot_key, spot_config in self.current_spots['surf_spots'].items():
            try:
                surf_spot = {
                    'name': spot_config.get('name', spot_key),
                    'latitude': float(spot_config.get('latitude', '0.0')),
                    'longitude': float(spot_config.get('longitude', '0.0')),
                    'beach_angle': float(spot_config.get('beach_facing', '270.0')),
                    'bottom_type': spot_config.get('bottom_type', 'sand'),
                    'exposure': spot_config.get('exposure', 'exposed'),
                    'bathymetry_calculated': spot_config.get('bathymetry_calculated', 'false') == 'true'
                }
                result['surf_spots'].append(surf_spot)
            except ValueError:
                # Skip spots with invalid coordinates
                continue
        
        # Convert fishing spots
        for spot_key, spot_config in self.current_spots['fishing_spots'].items():
            try:
                fishing_spot = {
                    'name': spot_config.get('name', spot_key),
                    'latitude': float(spot_config.get('latitude', '0.0')),
                    'longitude': float(spot_config.get('longitude', '0.0')),
                    'location_type': spot_config.get('location_type', 'shore'),
                    'target_category': spot_config.get('target_category', 'mixed_bag')
                }
                result['fishing_spots'].append(fishing_spot)
            except ValueError:
                # Skip spots with invalid coordinates
                continue
        
        return result

    def _get_surf_spot_validation_status(self, spot_config):
        """Get detailed validation status for a surf spot"""
        
        # For now, simulate depth check - in real implementation this would call GEBCO API
        lat = float(spot_config.get('latitude', 0))
        depth = abs(lat - 30) * 0.3 + 2.0  # Mock depth calculation
        
        # Load surf depth configuration directly from yaml_data
        try:
            bathymetry_config = self.yaml_data.get('bathymetry_data', {})
            surf_validation = bathymetry_config.get('surf_break_validation', {})
            depth_range = surf_validation.get('optimal_depth_range', {})
            optimal_min = depth_range.get('min_meters', 1.5)
            optimal_max = depth_range.get('max_meters', 6.0)
        except:
            # Fallback values if YAML config fails
            optimal_min, optimal_max = 1.5, 6.0
        
        if optimal_min <= depth <= optimal_max:
            return {
                'has_issue': False,
                'icon': '✅',
                'display_text': f'Optimal: {depth:.1f}m',
                'depth': depth,
                'recommendation': None
            }
        elif depth < config['shallow_limit']:
            return {
                'has_issue': True,
                'icon': '⚠️',
                'display_text': f'Too shallow: {depth:.1f}m',
                'depth': depth,
                'recommendation': 'Move coordinates seaward (away from shore)'
            }
        else:
            return {
                'has_issue': True,
                'icon': '⚠️', 
                'display_text': f'Too deep: {depth:.1f}m',
                'depth': depth,
                'recommendation': 'Move coordinates shoreward (toward beach)'
            }

    def _show_validation_info_dialog(self, stdscr, spot_type: str, spot_item):
        """Show validation info dialog for surf spots"""
        
        spot_key, spot_config = spot_item
        validation_status = self._get_surf_spot_validation_status(spot_config)
        
        height, width = stdscr.getmaxyx()
        
        dialog_height = 10
        dialog_width = 60
        dialog_y = (height - dialog_height) // 2
        dialog_x = (width - dialog_width) // 2
        
        dialog_win = curses.newwin(dialog_height, dialog_width, dialog_y, dialog_x)
        dialog_win.box()
        
        name = spot_config.get('name', 'Unknown')
        dialog_win.addstr(1, 2, f"Depth Info: {name}", curses.A_BOLD)
        
        dialog_win.addstr(3, 2, f"Current depth: {validation_status['depth']:.1f} meters")
        dialog_win.addstr(4, 2, f"Optimal range: 1.5m - 6.0m")
        dialog_win.addstr(5, 2, f"Status: {validation_status['display_text']}")
        
        if validation_status['recommendation']:
            dialog_win.addstr(6, 2, f"Fix: {validation_status['recommendation']}")
            dialog_win.addstr(7, 2, "Use NOAA Bathymetric Viewer for better coordinates")
        
        dialog_win.addstr(8, 2, "Press any key to continue...", curses.color_pair(5))
        dialog_win.refresh()
        
        dialog_win.getch()
        return 'continue'    

class GRIBLibraryManager:
    """Detect GRIB processing libraries and enforce prerequisites"""
    
    def __init__(self):
        self.available_library = None
    
    def detect_grib_libraries(self):
        """Detect available GRIB processing libraries"""
        
        # Method 1: Try eccodes-python (preferred)
        try:
            import eccodes
            self.available_library = 'eccodes'
            return True
        except ImportError:
            pass
        
        # Method 2: Try pygrib (fallback)
        try:
            import pygrib
            self.available_library = 'pygrib'
            return True
        except ImportError:
            pass
        
        return False
    
    def install_grib_library(self):
        """Check GRIB library availability and exit if not found"""
        
        print(f"\n{CORE_ICONS['selection']} GRIB Processing Library Check")
        print("GFS Wave forecast data requires GRIB file processing capability.")
        print()
        
        if self.detect_grib_libraries():
            print(f"  {CORE_ICONS['status']} GRIB library detected: {self.available_library}")
            return True
        else:
            print(f"  {CORE_ICONS['warning']} No GRIB library found")
            print()
            print("PREREQUISITE MISSING:")
            print("This extension requires pygrib or eccodes-python to process")
            print("GFS Wave GRIB forecast data for beach-specific surf forecasts.")
            print()
            print("Installation options:")
            print("  1. Recommended (lightweight - 161MB):")
            print("     sudo apt-get install python3-grib")
            print("  2. Alternative (heavier - 500MB+):")
            print("     sudo apt-get install libeccodes0 python3-eccodes")
            print("  3. pip fallback:")
            print("     pip install pygrib")
            print()
            print("The python3-grib package provides pygrib with reasonable dependencies.")
            print("This enables 16km resolution GFS Wave forecasts for your surf spots.")
            print()
            print("Please install a GRIB library and run the installer again.")
            print("See README.md for detailed installation instructions.")
            sys.exit(1)


class GEBCOAPIClient:
    """GEBCO Bathymetry API client for installation-time depth queries"""
    
    def __init__(self, yaml_data):
        """Initialize GEBCO API client with configuration from YAML"""
        self.bathymetry_config = yaml_data.get('bathymetry_data', {})
        self.api_config = self.bathymetry_config.get('api_configuration', {})
        self.fallback_config = self.bathymetry_config.get('fallback_configuration', {})
        self.validation_config = self.bathymetry_config.get('validation_thresholds', {})
        
        # Data-driven API settings from YAML
        self.base_url = self.api_config.get('base_url')
        self.timeout = self.api_config.get('timeout_seconds')
        self.retry_attempts = self.api_config.get('retry_attempts')
        self.rate_limit_delay = self.api_config.get('rate_limit_delay')
        
    def query_bathymetry_with_fallback(self, coordinates_list, progress_manager=None):
        """
        Query GEBCO API for bathymetry data with user-controlled fallback
        Returns: (success, bathymetry_data, used_fallback)
        """
        
        if progress_manager:
            progress_manager.show_step_progress("Querying GEBCO bathymetry API")
        
        # Attempt API query first
        success, bathymetry_data = self._query_gebco_api(coordinates_list)
        
        if progress_manager:
            if success:
                progress_manager.complete_step("GEBCO bathymetry retrieved")
            else:
                progress_manager.show_error("GEBCO API query", "API unavailable")
        
        if success:
            return True, bathymetry_data, False
        
        # API failed - present user with fallback options
        return self._handle_gebco_fallback(coordinates_list)
    
    def _query_gebco_api(self, coordinates_list):
        """Query GEBCO API for list of coordinates"""
        
        try:
            # Format coordinates for API request
            locations_param = '|'.join([f"{lat},{lon}" for lat, lon in coordinates_list])
            api_config = self.bathymetry_config.get('api_configuration', {})
            interpolation = api_config.get('interpolation', 'bilinear')  # Default fallback
            
            url = f"{self.base_url}?locations={locations_param}&interpolation={interpolation}"
            
            print(f"DEBUG: GEBCO URL = {url}")
            print(f"DEBUG: coordinates_list = {coordinates_list}")
            print(f"DEBUG: locations_param = {locations_param}")
            print(f"DEBUG: interpolation = {interpolation}")
            print(f"DEBUG: base_url = {self.base_url}")

            # Make API request with timeout
            request = urllib.request.Request(url)
            request.add_header('User-Agent', 'WeeWX-SurfFishing/2.0')
            
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
            
            if data.get('status') == 'OK':
                bathymetry_data = []
                for result in data.get('results', []):
                    elevation = result.get('elevation', 0.0)
                    depth = abs(elevation) if elevation < 0 else 0.0  # GEBCO uses negative for water depth
                    bathymetry_data.append(depth)
                
                return True, bathymetry_data
            else:
                return False, None
                
        except Exception as e:
            print(f"  {CORE_ICONS['warning']} GEBCO API Error: {e}")
            return False, None
    
    def _handle_gebco_fallback(self, coordinates_list):
        """Handle GEBCO API failure with user choice"""
        
        if not self.fallback_config.get('require_user_consent'):
            return False, None, False
        
        # Display warning and get user choice
        print(f"\n{CORE_ICONS['warning']} GEBCO BATHYMETRY API UNAVAILABLE")
        print("="*60)
        
        accuracy_impact = self.fallback_config.get('accuracy_impact_estimate')
        print(f"Impact: Reduced surf forecast accuracy ({accuracy_impact})")
        print("Affected Features: Wave shoaling, refraction, and breaking predictions")
        print()
        
        print("Please choose how to proceed:")
        print("1. CONTINUE WITH DEFAULTS - Install now using generic depth profiles")
        print(f"   {CORE_ICONS['status']} Installation completes immediately")
        print(f"   {CORE_ICONS['warning']} Reduced forecast accuracy until GEBCO data added later")
        print()
        print("2. RETRY GEBCO API - Attempt to connect again")
        print(f"   {CORE_ICONS['status']} May resolve temporary network issues")
        print(f"   {CORE_ICONS['warning']} May fail again if service is down")
        print()
        print("3. ABORT INSTALLATION - Exit and try again later")
        print(f"   {CORE_ICONS['status']} Ensures best possible forecast accuracy")
        print(f"   {CORE_ICONS['warning']} Must reinstall when GEBCO service available")
        print()
        
        while True:
            choice = input("Choice [1/2/3]: ").strip()
            
            if choice == '1':
                # User chose to proceed with defaults
                if self._get_fallback_consent():
                    default_depths = self.fallback_config.get('default_path_depths')
                    # Create fallback data for all coordinate sets
                    fallback_data = [default_depths[:len(coordinates_list)] for _ in range(len(coordinates_list))]
                    return True, fallback_data, True
                else:
                    continue  # User declined, ask again
                    
            elif choice == '2':
                # Retry API
                print(f"  Retrying GEBCO API...")
                time.sleep(2)  # Brief delay before retry
                success, bathymetry_data = self._query_gebco_api(coordinates_list)
                if success:
                    return True, bathymetry_data, False
                else:
                    print(f"  {CORE_ICONS['warning']} Retry failed. API still unavailable.")
                    continue  # Go back to choice menu
                    
            elif choice == '3':
                # Abort installation
                print("\nInstallation aborted. Please try again when GEBCO API is available.")
                sys.exit(0)
                
            else:
                print(f"{CORE_ICONS['warning']} Please enter 1, 2, or 3")
    
    def _get_fallback_consent(self):
        """Get explicit user consent for using fallback data"""
        
        if not self.fallback_config.get('show_accuracy_warning'):
            return True
        
        print(f"\n{CORE_ICONS['warning']} ACCURACY WARNING")
        print("Using default depth values will reduce surf forecast precision.")
        print("Your forecasts will be functional but less accurate than with real bathymetry.")
        print()
        print("NOTE: GEBCO API can be intermittently unavailable due to maintenance or high usage.")
        print("You can try installing again later when the service is restored, or proceed with")
        print("reduced accuracy now and update bathymetry data later via extension reconfiguration.")
        print()
        
        while True:
            consent = input('Type "CONFIRM" to proceed with reduced accuracy: ').strip().upper()
            if consent == "CONFIRM":
                print(f"\n{CORE_ICONS['status']} Proceeding with default bathymetry values")
                return True
            elif consent.upper() in ['NO', 'N', 'CANCEL', 'ABORT']:
                return False
            else:
                print('Please type "CONFIRM" to proceed or "NO" to cancel')


class AtmosphericDataAnalyzer:
    """Analyze atmospheric data coverage for user guidance"""
    
    def __init__(self, phase_i_data):
        self.phase_i_data = phase_i_data
    
    def analyze_atmospheric_coverage(self, surf_spots, ndbc_stations):
        """Analyze atmospheric data coverage for user's spots"""
        
        analysis = {
            'total_spots': len(surf_spots),
            'buoy_covered_spots': 0,
            'station_recommended_spots': 0,
            'coverage_details': [],
            'station_recommended': False,
            'hybrid_recommended': False
        }
        
        for spot in surf_spots:
            spot_analysis = {
                'name': spot['name'],
                'lat': spot['latitude'],
                'lon': spot['longitude']
            }
            
            # Find nearest atmospheric NDBC buoy
            nearest_atmospheric = self._find_nearest_atmospheric_buoy(spot, ndbc_stations)
            
            if nearest_atmospheric and nearest_atmospheric['distance'] <= 25:
                spot_analysis['wind_coverage'] = 'buoy'
                spot_analysis['buoy_distance'] = nearest_atmospheric['distance']
                spot_analysis['buoy_id'] = nearest_atmospheric['id']
                analysis['buoy_covered_spots'] += 1
            else:
                spot_analysis['wind_coverage'] = 'station_recommended'
                spot_analysis['reason'] = 'No atmospheric buoys within 25 miles'
                analysis['station_recommended_spots'] += 1
            
            analysis['coverage_details'].append(spot_analysis)
        
        # Determine overall recommendation
        if analysis['station_recommended_spots'] > analysis['buoy_covered_spots']:
            analysis['station_recommended'] = True
        else:
            analysis['hybrid_recommended'] = True
        
        return analysis
    
    def _find_nearest_atmospheric_buoy(self, spot, ndbc_stations):
        """Find nearest NDBC buoy with atmospheric sensors"""
        
        atmospheric_buoys = []
        
        for station in ndbc_stations:
            # Check if buoy has atmospheric capabilities
            if self._has_atmospheric_sensors(station):
                distance = self._calculate_distance(
                    spot['latitude'], spot['longitude'],
                    station['lat'], station['lon']
                )
                atmospheric_buoys.append({
                    'id': station['id'],
                    'distance': distance,
                    'lat': station['lat'],
                    'lon': station['lon']
                })
        
        if atmospheric_buoys:
            return min(atmospheric_buoys, key=lambda x: x['distance'])
        return None
    
    def _has_atmospheric_sensors(self, station):
        """Check if NDBC station has atmospheric sensors"""
        # This would check station metadata for wind/pressure capabilities
        # For now, approximate based on station type/name patterns
        station_name = station.get('name', '').lower()
        return any(keyword in station_name for keyword in [
            'weather', 'meteorological', 'met', 'atmospheric'
        ]) or station.get('type') == 'weather_buoy'
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in miles"""
        R = 3959  # Earth radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c


class SurfFishingConfigurator:
    """Interactive configuration for surf and fishing forecasts"""
    
    def __init__(self, config_dict, yaml_data):
        """Initialize configurator with GEBCO API client"""
        self.config_dict = config_dict
        self.yaml_data = yaml_data
        self.progress = InstallationProgressManager()
        self.grib_manager = GRIBLibraryManager()
        
        # NEW: Initialize GEBCO API client
        self.gebco_client = GEBCOAPIClient(yaml_data)

    def _convert_cardinal_to_degrees(self, cardinal_input):
        """Convert 16-point cardinal direction to degrees"""
        
        cardinal_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        
        return cardinal_map.get(cardinal_input.upper())

    def _get_beach_angle(self):
        """Get beach angle from user with validation"""
        
        print("  Beach Angle Configuration:")
        print("  Standing on the beach looking out to sea, what compass direction are you facing?")
        print()
        print("  Examples:")
        print("  - Malibu (facing south): S or 180")
        print("  - Ocean Beach SF (facing west): W or 270")
        print("  - Cape Cod (facing east): E or 90")
        print()
        print("  Enter as cardinal direction (N/NE/E/SE/S/SW/W/NW/etc) or degrees (0-360):")
        
        while True:
            angle_input = input("  Beach angle: ").strip()
            
            if not angle_input:
                print(f"  {CORE_ICONS['warning']} Please enter a beach angle")
                continue
            
            # Try to parse as numeric degrees first
            try:
                degrees = float(angle_input)
                if 0 <= degrees <= 360:
                    return degrees
                else:
                    print(f"  {CORE_ICONS['warning']} Degrees must be between 0 and 360")
                    continue
            except ValueError:
                pass
            
            # Try to parse as cardinal direction
            degrees = self._convert_cardinal_to_degrees(angle_input)
            if degrees is not None:
                return degrees
            
            print(f"  {CORE_ICONS['warning']} Invalid input. Use cardinal directions (N, NE, etc) or degrees (0-360)")

    def _validate_water_location(self, lat, lon, beach_angle=None, spot_name=None):
            """
            Combined water location validation and surf break depth adjustment using single GEBCO API call
            """
            validation_config = self.yaml_data.get('bathymetry_data', {}).get('coordinate_validation', {})
            if not validation_config.get('enable_land_sea_validation'):
                if beach_angle is not None and spot_name is not None:
                    return lat, lon  # Return coordinates for surf spot usage
                return True  # Return boolean for simple water validation
            
            self.progress.show_step_progress("Validating water location")
            
            # Single GEBCO query for both land/sea and depth validations
            success, bathymetry_data, used_fallback = self.gebco_client.query_bathymetry_with_fallback([(lat, lon)], self.progress)
            
            if not success:
                self.progress.show_error("Location validation", "Cannot validate - proceeding anyway")
                if beach_angle is not None and spot_name is not None:
                    return lat, lon  # Return coordinates for surf spot usage
                return True  # Return boolean for simple water validation
            
            depth = bathymetry_data[0] if bathymetry_data else 0.0
            land_threshold = validation_config.get('land_elevation_threshold', 0.0)
            
            # First validation: Land/sea check
            if depth <= land_threshold:
                self.progress.show_error("Location validation", f"Location on land (elevation: {depth:.1f}m)")
                if beach_angle is not None and spot_name is not None:
                    return None, None  # Invalid location for surf spots
                return False  # Invalid for simple water validation
            
            self.progress.complete_step(f"Valid water location (depth: {depth:.1f}m)")
            
            # If this is a surf spot validation (beach_angle and spot_name provided), do surf depth validation
            if beach_angle is not None and spot_name is not None:
                print(f"  {CORE_ICONS['selection']} Validating surf break depth for optimal conditions...")
                
                # Load surf depth configuration from YAML
                config = self._load_surf_depth_config()
                optimal_min, optimal_max = config['optimal_min'], config['optimal_max']
                
                # Perfect depth - no adjustment needed
                if optimal_min <= depth <= optimal_max:
                    print(f"  {CORE_ICONS['status']} Perfect surf break depth: {depth:.1f}m")
                    return lat, lon
                
                # Too shallow - mandatory adjustment (safety)
                elif depth < config['shallow_limit']:
                    print(f"  {CORE_ICONS['warning']} Location too shallow: {depth:.1f}m")
                    print(f"  Adjusting seaward to find suitable surf break depth...")
                    
                    adjusted_lat, adjusted_lon, adjusted_depth = self._try_coordinate_adjustment(
                        lat, lon, beach_angle, depth, direction='seaward', spot_name=spot_name, config=config)
                    
                    if adjusted_lat is not None:
                        distance_moved = self._calculate_distance(lat, lon, adjusted_lat, adjusted_lon) * 1609.34  # Convert to meters
                        print(f"  {CORE_ICONS['status']} Adjusted {distance_moved:.0f}m seaward to {adjusted_depth:.1f}m depth")
                        return adjusted_lat, adjusted_lon
                    else:
                        return self._handle_adjustment_failure(lat, lon, depth, spot_name, "shallow")
                
                # Too deep - user choice
                elif depth > config['deep_warning']:
                    print(f"  {CORE_ICONS['warning']} Depth Analysis: Current location depth is {depth:.1f}m")
                    print(f"  This is deeper than typical surf break range ({optimal_min}-{optimal_max}m).")
                    print(f"  This could be suitable for big wave conditions, or you may want")
                    print(f"  to move closer to shore for regular surf conditions.")
                    print()
                    print("  Options:")
                    print("  1. Keep current location (suitable for big waves)")
                    print("  2. Auto-adjust toward shore (find typical surf depth)")  
                    print("  3. Enter new coordinates")
                    
                    while True:
                        choice = input("  Choice (1-3): ").strip()
                        if choice == '1':
                            print(f"  {CORE_ICONS['status']} Using deep water location ({depth:.1f}m)")
                            return lat, lon
                        elif choice == '2':
                            print(f"  Adjusting shoreward to find suitable surf break depth...")
                            adjusted_lat, adjusted_lon, adjusted_depth = self._try_coordinate_adjustment(
                                lat, lon, beach_angle, depth, direction='shoreward', spot_name=spot_name, config=config)
                            
                            if adjusted_lat is not None:
                                distance_moved = self._calculate_distance(lat, lon, adjusted_lat, adjusted_lon) * 1609.34
                                print(f"  {CORE_ICONS['status']} Adjusted {distance_moved:.0f}m shoreward to {adjusted_depth:.1f}m depth")
                                return adjusted_lat, adjusted_lon
                            else:
                                return self._handle_adjustment_failure(lat, lon, depth, spot_name, "deep")
                        elif choice == '3':
                            print(f"  Please enter new coordinates for {spot_name}")
                            return self._get_coordinates_for_water_location("surf break")
                        else:
                            print(f"  {CORE_ICONS['warning']} Please enter 1, 2, or 3")
                
                return lat, lon
            
            # For simple water validation, just return True
            return True
       
    def run_interactive_setup(self):
        """Main configuration workflow with cursor UI integration"""
        
        print(f"{CORE_ICONS['navigation']} Surf & Fishing Forecast Configuration")
        print("="*60)
        print("Configure your personal surf and fishing forecast system")
        print("This extension reads data from Phase I and adds forecasting capabilities")
        print()
        
        # Step 1: Check Phase I dependency (UNCHANGED)
        self._check_phase_i_dependency()
        
        # Step 2: Setup GRIB processing libraries (UNCHANGED) 
        grib_available = self._setup_grib_processing()
        
        # Step 3: Configure data source strategy (UNCHANGED)
        data_sources = self._configure_data_sources()
        
        # Step 4: Configure forecast types and locations (ENHANCED)
        forecast_types = self._select_location_types()
        
        # NEW: Check for existing configuration
        existing_spots = self._check_existing_spots_in_conf()
        total_existing = existing_spots['surf_count'] + existing_spots['fishing_count']
        
        if total_existing > 0:
            print(f"\n{CORE_ICONS['status']} Found existing configuration:")
            print(f"  • {existing_spots['surf_count']} surf spots")
            print(f"  • {existing_spots['fishing_count']} fishing spots")
            print()
            
            while True:
                choice = input("Use enhanced management interface for existing spots? (y/n, default y): ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    use_cursor_ui = True
                    break
                elif choice in ['n', 'no']:
                    use_cursor_ui = False
                    break
                else:
                    print(f"{CORE_ICONS['warning']} Please enter y or n")
        else:
            print(f"\n{CORE_ICONS['selection']} No existing spots found - starting fresh configuration")
            while True:
                choice = input("Use enhanced cursor UI interface? (y/n, default y): ").strip().lower()
                if choice in ['', 'y', 'yes']:
                    use_cursor_ui = True
                    break
                elif choice in ['n', 'no']:
                    use_cursor_ui = False
                    break
                else:
                    print(f"{CORE_ICONS['warning']} Please enter y or n")
        
        # Configure locations with chosen method
        if use_cursor_ui:
            selected_locations = self._configure_locations_with_cursor_ui(forecast_types)
        else:
            selected_locations = self._configure_locations_manually(forecast_types)
        
        # Step 5: Analyze marine station integration (UNCHANGED)
        station_analysis = self._analyze_marine_station_integration(selected_locations)
        
        # Step 6: Create configuration dictionary (UNCHANGED)
        config_dict = self._create_config_dict(forecast_types, data_sources, selected_locations, grib_available, station_analysis)
        
        return config_dict, selected_locations
    
    def _check_phase_i_dependency(self):
        """Verify Phase I marine data extension is installed"""
        
        print(f"{CORE_ICONS['selection']} Checking Phase I Dependency")
        
        # Check if Phase I service is configured
        if 'MarineDataService' not in self.config_dict:
            print(f"  {CORE_ICONS['warning']} Phase I Marine Data Extension not found")
            print("  This extension requires Phase I to be installed first")
            print("  Please install the Phase I marine data extension before continuing")
            sys.exit(1)
        
        print(f"  {CORE_ICONS['status']} Phase I Marine Data Extension detected")
    
    def _setup_grib_processing(self):
        """Setup GRIB processing libraries"""
        
        if self.grib_manager.detect_grib_libraries():
            print(f"  {CORE_ICONS['status']} GRIB library detected: {self.grib_manager.available_library}")
            return True
        else:
            return self.grib_manager.install_grib_library()
    
    def _configure_data_sources(self):
        """
        Configure atmospheric data strategy based on user's station location
        Uses simple question about coastal proximity for decision making
        """
        
        print(f"\n{CORE_ICONS['selection']} Station Location Assessment")
        print("To optimize your surf and fishing forecasts, we need to understand your station location.")
        print()
        
        # Get station coordinates for display (if available)
        station_lat = self.config_dict.get('Station', {}).get('latitude')
        station_lon = self.config_dict.get('Station', {}).get('longitude')
        
        if station_lat and station_lon:
            print(f"  Your WeeWX Station: {float(station_lat):.4f}, {float(station_lon):.4f}")
        
        print(f"  Location Question:")
        print()
        print("Is your weather station located within 5 miles of the ocean/coast?")
        print("(This determines whether your station data is useful for marine forecasting)")
        print()
        print("1. Yes - My station is within 5 miles of the coast")
        print("   → We'll use your station + NOAA data for best accuracy")
        print()
        print("2. No - My station is more than 5 miles inland") 
        print("   → We'll use NOAA marine data only for best accuracy")
        print()
        
        while True:
            try:
                choice = input("Is your station within 5 miles of the coast? (1=Yes, 2=No): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nInstallation cancelled by user.")
                sys.exit(1)
            
            if choice == '1':
                print(f"\n  {CORE_ICONS['status']} DECISION: Hybrid Approach (Station + NOAA)")
                print(f"  Your coastal station will provide valuable local atmospheric data")
                print(f"      - Station: Local wind/pressure accuracy")
                print(f"      - NOAA: Marine wave/tide data")
                print(f"      - Combined: Best forecast accuracy")
                
                # Call existing station integration method
                return self._configure_station_integration()
                
            elif choice == '2':
                print(f"\n  {CORE_ICONS['status']} DECISION: NOAA-Only Approach")
                print(f"  NOAA marine buoys will provide the most accurate data")
                print(f"      - NOAA buoys: Best marine atmospheric data")
                print(f"      - Station: Too far inland for marine forecasts")
                
                return {
                    'type': 'noaa_only',
                    'reason': 'station_inland_user_selected'
                }
            else:
                print(f"  {CORE_ICONS['warning']} Please enter 1 for Yes or 2 for No")
    
    def _configure_station_integration(self):
        """
        Configure which station sensors to use (supplement with NOAA)
        Enhanced with location-based recommendations for coastal stations
        """
        
        print(f"\n{CORE_ICONS['selection']} Station Sensor Integration")
        print("Select which station sensors to use (supplement with NOAA):")
        print("Recommendations for coastal stations:")
        print()
        
        # Coastal station recommendations (since user selected coastal location)
        recommendations = {
            'wind': 'Highly recommended - excellent coastal wind representation',
            'pressure': 'Highly recommended - excellent for marine pressure systems', 
            'temperature': 'Recommended - good coastal air temperature'
        }
        
        available_sensors = {
            'wind': 'Wind speed and direction',
            'pressure': 'Barometric pressure', 
            'temperature': 'Air temperature'
        }
        
        selected_sensors = {}
        for sensor_key, description in available_sensors.items():
            recommendation = recommendations.get(sensor_key, 'Recommended')
            use_sensor = input(f"Use station {description}? ({recommendation}) (y/n, default n): ").strip().lower()
            selected_sensors[sensor_key] = use_sensor == 'y'
            
        return {
            'type': 'station_supplement',
            'sensors': selected_sensors,
            'reason': 'station_coastal_user_selected'
        }
    
    def _configure_locations(self):
        """Configure surf and fishing spots"""
        
        print(f"\n{CORE_ICONS['navigation']} Location Configuration")
        print("="*60)
        print("Configure your surf and fishing forecast locations")
        print("Maximum 10 locations total (surf + fishing combined)")
        print()
        
        locations = {
            'surf_spots': [],
            'fishing_spots': []
        }
        
        # Choose location types to configure
        location_types = self._select_location_types()
        
        # Configure each type
        if 'surf' in location_types:
            locations['surf_spots'] = self._configure_surf_spots()
            
        if 'fishing' in location_types:
            locations['fishing_spots'] = self._configure_fishing_spots()
        
        return locations
    
    def _select_location_types(self):
        """Select which types of locations to configure"""
        
        print("Select forecast types to configure:")
        print("1. Surf forecasts only")
        print("2. Fishing forecasts only") 
        print("3. Both surf and fishing forecasts")
        print()
        
        while True:
            choice = input("Select option (1-3): ").strip()
            if choice == '1':
                return ['surf']
            elif choice == '2':
                return ['fishing']
            elif choice == '3':
                return ['surf', 'fishing']
            else:
                print(f"{CORE_ICONS['warning']} Please enter 1, 2, or 3")
    
    def _configure_surf_spots(self):
        """
        Configure surf spots with simplified approach removing complex bathymetry calculations
        """
        surf_spots = []
        print(f"\n{CORE_ICONS['selection']} Surf Spot Configuration")
        print("Configure your surf spots with enhanced depth validation")
        print()
        
        spot_count = 1
        while True:
            print(f"Surf Spot {spot_count}:")
            
            name = input("  Spot name (e.g., 'Malibu', 'Ocean Beach') [Enter to finish]: ").strip()
            if not name:
                break
                
            # Get initial coordinates for surf break
            lat, lon = self._get_coordinates_with_validation("surf break")
            
            # Get beach angle
            beach_angle = self._get_beach_angle()
            
            # Combined water validation and surf depth adjustment (single GEBCO call)
            lat, lon = self._validate_water_location(lat, lon, beach_angle, name)
            
            # Check if validation failed
            if lat is None or lon is None:
                print(f"  {CORE_ICONS['warning']} Location validation failed - please try different coordinates")
                continue
            
            # Get surf characteristics (KEEP existing functionality)
            spot_config = self._configure_surf_characteristics(name, lat, lon)
            
            # SIMPLIFIED: Create spot data with service coordination flag
            spot_data = {
                'name': name,
                'latitude': lat,
                'longitude': lon,
                'beach_angle': beach_angle,
                'bottom_type': spot_config['bottom_type'],
                'exposure': spot_config['exposure'],
                'bathymetry_calculated': False  # Service will handle complex calculations
            }
            
            surf_spots.append(spot_data)
            print(f"  {CORE_ICONS['status']} Added {name}")
            print()
            spot_count += 1
        
        return surf_spots
    
    def _configure_fishing_spots(self):
        """Configure fishing-specific locations with land/sea validation"""
        
        fishing_spots = []
        print(f"\n{CORE_ICONS['selection']} Fishing Spot Configuration")
        print("Enter your fishing spots")
        print()
        
        spot_count = 1
        while True:
            print(f"Fishing Spot {spot_count}:")
            
            name = input("  Spot name (e.g., 'Pier 39', 'Half Moon Bay') [Enter to finish]: ").strip()
            if not name:
                break
            
            # Get coordinates with land/sea validation
            lat, lon = self._get_coordinates_with_validation("fishing spot")
            
            location_config = self._configure_fishing_characteristics(name, lat, lon)
            
            spot_data = {
                'name': name,
                'latitude': lat,
                'longitude': lon,
                'location_type': location_config.get('location_type', 'shore')
            }
            
            fishing_spots.append(spot_data)
            print(f"  {CORE_ICONS['status']} Added {name}")
            print()
            spot_count += 1
        
        return fishing_spots
    
    def _get_coordinates_with_validation(self, location_name):
        """Get and validate coordinates for a location"""
        
        print(f"  Coordinates for {location_name}:")
        
        try:
            lat_str = input("    Latitude (decimal degrees, e.g., 34.0522): ").strip()
            lon_str = input("    Longitude (decimal degrees, e.g., -118.2437): ").strip()
            
            lat = float(lat_str)
            lon = float(lon_str)
            
            # Basic validation for US coastal waters
            if not (-180 <= lon <= 180):
                print(f"    {CORE_ICONS['warning']} Invalid longitude. Must be between -180 and 180")
                return None, None
                
            if not (-90 <= lat <= 90):
                print(f"    {CORE_ICONS['warning']} Invalid latitude. Must be between -90 and 90")
                return None, None
            
            return lat, lon
            
        except ValueError:
            print(f"    {CORE_ICONS['warning']} Invalid coordinates. Please enter decimal numbers")
            return None, None
    
    def _configure_surf_characteristics(self, name, lat, lon):
        """Configure surf-specific spot characteristics"""
        
        config = {
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'type': 'surf'
        }
        
        # Bottom type affects wave breaking patterns
        print("  Bottom type (affects how waves break):")
        print("    1. Sand (Beach break)")
        print("    2. Reef (Coral/rock reef)")
        print("    3. Point break (Rocky point)")
        print("    4. Jetty/Pier (Man-made structure)")
        print("    5. Mixed")
        
        bottom_choice = input("  Select bottom type (1-5, default 1): ").strip() or "1"
        config['bottom_type'] = {
            "1": "sand", "2": "reef", "3": "point", "4": "jetty", "5": "mixed"
        }.get(bottom_choice, "sand")
        
        # Exposure to swell
        print("  Exposure to ocean swell:")
        print("    1. Fully exposed (open ocean)")
        print("    2. Semi-protected (bay/cove)")
        print("    3. Protected (harbor/inlet)")
        
        exposure_choice = input("  Select exposure (1-3, default 1): ").strip() or "1"
        config['exposure'] = {
            "1": "exposed", "2": "semi_protected", "3": "protected"
        }.get(exposure_choice, "exposed")
        
        return config
    
    def _configure_fishing_characteristics(self, name, lat, lon):
        """Configure fishing-specific spot characteristics"""
        
        config = {
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'type': 'fishing'
        }
        
        # Location type affects fishing methods
        print("  Fishing location type:")
        print("    1. Shore/Beach (fishing from land)")
        print("    2. Pier/Jetty (fishing from structure)")
        print("    3. Boat (fishing from vessel)")
        print("    4. Mixed (multiple methods)")
        
        location_choice = input("  Select location type (1-4, default 1): ").strip() or "1"
        config['location_type'] = {
            "1": "shore", "2": "pier", "3": "boat", "4": "mixed"
        }.get(location_choice, "shore")
        
        # Target species category
        fish_categories = self.yaml_data.get('fish_categories', {})
        
        print("  Primary target fish category:")
        category_keys = list(fish_categories.keys())
        for i, (category_key, category_data) in enumerate(fish_categories.items(), 1):
            print(f"    {i}. {category_data['display_name']}")
        
        category_choice = input(f"  Select category (1-{len(fish_categories)}, default 1): ").strip() or "1"
        try:
            selected_category = category_keys[int(category_choice)-1]
        except (ValueError, IndexError):
            selected_category = category_keys[0] if category_keys else 'saltwater_inshore'  
        
        config['target_category'] = selected_category
        
        return config
    
    def _create_config_dict(self, forecast_types, data_sources, selected_locations, grib_available, station_analysis):
            """
            Create simplified configuration dictionary removing complex bathymetric storage and adding service coordination
            """
            
            # Base configuration structure - UNCHANGED
            config_dict = {
                'SurfFishingService': {
                    'enable': 'true',
                    'forecast_interval': '21600',  # 6 hours in seconds
                    'log_success': 'false',
                    'log_errors': 'true',
                    'timeout': '60',
                    'retry_attempts': '3',
                    
                    # Forecast configuration based on user selections - UNCHANGED
                    'forecast_settings': {
                        'enabled_types': ','.join(forecast_types),
                        'forecast_hours': '72',  # 3-day forecasts
                        'rating_system': 'five_star',
                        'update_interval_hours': '6'
                    },
                    
                    # Data source configuration - UNCHANGED
                    'data_integration': {
                        'method': data_sources.get('type', 'noaa_only'),
                        'local_station_distance_km': str(station_analysis.get('distance_km', 999) if station_analysis else 999),
                        'enable_station_data': 'true' if data_sources.get('type') == 'station_supplement' else 'false'
                    },
                    
                    # GRIB processing configuration - UNCHANGED
                    'grib_processing': {
                        'available': 'true' if grib_available else 'false',
                        'library': 'pygrib' if grib_available else 'none'
                    },
                    
                    # Data sources configuration - UNCHANGED
                    'data_sources': {
                        'gfs_wave': {
                            'api_source': data_sources.get('api_source', 'gfs_wave'),
                            'grid_selected': data_sources.get('grid_selected', 'global.0p16'),
                            'collection_interval': str(data_sources.get('collection_interval', 10800))
                        }
                    }
                }
            }
            
            # Process YAML sections (exclude geographic_regions - install-time only)
            for section_name, section_data in self.yaml_data.items():
                if section_name in ['noaa_gfs_wave', 'bathymetry_data', 'fish_categories', 'scoring_criteria']:
                    # Convert section using enhanced method that excludes geographic boundaries
                    converted_section = self._convert_yaml_section_to_conf(section_data)
                    config_dict['SurfFishingService'][section_name] = converted_section
                elif section_name == 'geographic_regions':
                    # Skip - this is install-time only data
                    print(f"  {CORE_ICONS['navigation']} Skipping geographic_regions section (install-time only)")
                    continue
            
            # SIMPLIFIED: Add user surf spots with basic configuration only
            if 'surf_spots' in selected_locations and selected_locations['surf_spots']:
                config_dict['SurfFishingService']['surf_spots'] = {}
                
                for i, spot in enumerate(selected_locations['surf_spots']):
                    spot_key = f'spot_{i}'
                    
                    # SIMPLIFIED: Basic spot configuration for service coordination
                    spot_config = {
                        'name': spot['name'],
                        'latitude': str(spot['latitude']),
                        'longitude': str(spot['longitude']),
                        'beach_facing': str(spot['beach_angle']),
                        'bottom_type': spot.get('bottom_type', 'sand'),
                        'exposure': spot.get('exposure', 'exposed'),
                        'bathymetry_calculated': 'false',  # Service trigger flag
                        'active': 'true'
                    }
                    
                    # REMOVED: No more complex bathymetric_path data storage
                    # Service will add all bathymetric data when it performs calculations
                    
                    config_dict['SurfFishingService']['surf_spots'][spot_key] = spot_config
            
            # Add fishing spots - UNCHANGED
            if 'fishing_spots' in selected_locations and selected_locations['fishing_spots']:
                config_dict['SurfFishingService']['fishing_spots'] = {}
                
                for i, spot in enumerate(selected_locations['fishing_spots']):
                    spot_key = f'spot_{i}'
                    config_dict['SurfFishingService']['fishing_spots'][spot_key] = {
                        'name': spot['name'],
                        'latitude': str(spot['latitude']),
                        'longitude': str(spot['longitude']),
                        'location_type': spot.get('location_type', 'shore'),
                        'target_category': spot.get('target_category', 'saltwater_inshore'),
                        'active': 'true'
                    }
            
            # Add station analysis results - UNCHANGED
            if station_analysis:
                config_dict['SurfFishingService']['station_analysis'] = {
                    'analysis_completed': str(station_analysis.get('station_analysis_completed', False)),
                    'accepted_recommendations': str(len(station_analysis.get('accepted_recommendations', []))),
                    'coverage_quality': str(station_analysis.get('coverage_summary', {}).get('quality_score', 'unknown'))
                }
            
            return config_dict

    def _convert_yaml_section_to_conf(self, yaml_section):
        """Convert YAML section to CONF format - excludes geographic_regions from CONF"""
        
        # NEW: Skip geographic_regions section (install-time only data)
        if isinstance(yaml_section, dict) and 'geographic_regions' in yaml_section:
            print(f"  {CORE_ICONS['navigation']} Excluding geographic_regions from CONF (install-time only)")
            # Create filtered copy without geographic_regions
            yaml_section = {k: v for k, v in yaml_section.items() if k != 'geographic_regions'}
        
        # EXISTING: Handle dictionary conversion
        if isinstance(yaml_section, dict):
            conf_section = {}
            for key, value in yaml_section.items():
                # NEW: Skip geographic_regions at any nesting level
                if key == 'geographic_regions':
                    continue
                conf_section[key] = self._convert_yaml_section_to_conf(value)
            return conf_section
        
        # EXISTING: Handle list conversion
        elif isinstance(yaml_section, list):
            # Convert lists to comma-separated strings for CONF compatibility
            if all(isinstance(item, (str, int, float)) for item in yaml_section):
                return ','.join(str(item) for item in yaml_section)
            else:
                # For complex lists, convert to dictionary with string keys for CONF compatibility
                result = {}
                for i, item in enumerate(yaml_section):
                    result[str(i)] = self._convert_yaml_section_to_conf(item)  # Use str(i) instead of integer i
                return result
        
        # EXISTING: Convert primitives to string for CONF compatibility
        else:
            return str(yaml_section)
    
    def _analyze_marine_station_integration(self, selected_locations):
        """
        Analyze Phase I marine station coverage for user locations
        Provide recommendations for optimization if needed
        FIXED: Properly return coverage quality data for CONF
        """
        print(f"\n{CORE_ICONS['navigation']} Marine Station Integration Analysis")
        print("Analyzing your Phase I station coverage for optimal forecasting...")
        
        # Initialize analyzers
        phase_i_analyzer = PhaseIAnalyzer(self.config_dict)
        quality_analyzer = StationQualityAnalyzer(self.yaml_data)
        recommendation_engine = MarineStationRecommendationEngine(phase_i_analyzer, quality_analyzer)
        
        # Get Phase I coverage summary
        phase_i_coverage = phase_i_analyzer.analyze_phase_i_coverage()
        
        print(f"\nPhase I Station Summary:")
        print(f"  Total stations: {phase_i_coverage['total_stations']}")
        print(f"  CO-OPS tide stations: {phase_i_coverage['coops_count']}")
        print(f"  NDBC buoy stations: {phase_i_coverage['ndbc_count']}")
        
        if phase_i_coverage['distance_summary']['closest_station']:
            closest = phase_i_coverage['distance_summary']['closest_station']
            print(f"  Closest station: {closest['name']} ({closest['distance']:.1f} miles)")
        
        # Analyze all user locations
        all_user_locations = []
        for spot in selected_locations.get('surf_spots', []):
            all_user_locations.append(spot)
        for spot in selected_locations.get('fishing_spots', []):
            all_user_locations.append(spot)
        
        if not all_user_locations:
            print(f"\n{CORE_ICONS['status']} No location analysis needed - no locations configured")
            # FIXED: Return proper default analysis data
            return {
                'station_analysis_completed': True,
                'accepted_recommendations': [],
                'coverage_summary': {
                    'quality_score': 'no_locations',
                    'wave_quality': 0.0,
                    'atmospheric_quality': 0.0,
                    'tide_quality': 0.0
                }
            }
        
        # Generate recommendations
        recommendations = recommendation_engine.analyze_multi_location_optimization(all_user_locations)
        
        # Display current coverage summary
        current = recommendations['current_coverage']
        print(f"\nCoverage Quality Summary:")
        print(f"  Wave data quality: {current['wave_quality']:.1f}/1.0")
        print(f"  Atmospheric data quality: {current['atmospheric_quality']:.1f}/1.0")
        print(f"  Tide data quality: {current['tide_quality']:.1f}/1.0")
        
        # FIXED: Calculate overall quality score for CONF storage
        overall_quality = (current['wave_quality'] + current['atmospheric_quality'] + current['tide_quality']) / 3.0
        
        # Determine quality level for human-readable format
        if overall_quality >= 0.8:
            quality_level = 'excellent'
        elif overall_quality >= 0.6:
            quality_level = 'good'
        elif overall_quality >= 0.4:
            quality_level = 'fair'
        else:
            quality_level = 'poor'
        
        # Present recommendations interactively
        accepted_recommendations = recommendation_engine.display_recommendations_interactive(recommendations)
        
        if accepted_recommendations:
            print(f"\n{CORE_ICONS['selection']} Implementation Options")
            print("Accepted recommendations will be noted in your configuration.")
            print("You can implement these manually in Phase I after installation.")
        else:
            print(f"\nYour Phase I station configuration provides good coverage for all locations.")
            print("No immediate improvements needed.")
            print(f"{CORE_ICONS['status']} Current configuration will be used as-is")
        
        # FIXED: Return proper station analysis data with coverage quality
        return {
            'station_analysis_completed': True,
            'accepted_recommendations': accepted_recommendations or [],
            'coverage_summary': {
                'quality_score': quality_level,  # 'excellent', 'good', 'fair', 'poor'
                'quality_numeric': overall_quality,  # 0.0-1.0 numeric score
                'wave_quality': current['wave_quality'],
                'atmospheric_quality': current['atmospheric_quality'],
                'tide_quality': current['tide_quality'],
                'total_locations': current['total_locations']
            },
            'phase_i_summary': phase_i_coverage,
            'location_recommendations': recommendations
        }

    def _display_configuration_summary(self, config_dict, station_analysis):
            """
            Display comprehensive configuration summary including station analysis
            Provides clear overview of what was configured
            """
            print(f"\n{CORE_ICONS['status']} Configuration Summary")
            print("="*50)
            
            service_config = config_dict['SurfFishingService']
            
            # Forecast types configured
            forecast_types = service_config['forecast_settings']['enabled_types'].split(',') 
            print(f"Forecast Types: {', '.join(forecast_types).title()}")
            
            # Data sources
            data_source_type = service_config['data_integration']['method'] 
            print(f"Data Strategy: {data_source_type.replace('_', ' ').title()}")
            
            # Location counts
            surf_count = len(service_config.get('surf_spots', {}))
            fishing_count = len(service_config.get('fishing_spots', {})) 
            print(f"Locations: {surf_count} surf spots, {fishing_count} fishing spots")
            
            # Station integration analysis results (FIXED)
            if station_analysis and 'station_analysis' in service_config:
                analysis_config = service_config['station_analysis']
                print(f"\nStation Integration Analysis:")
                print(f"  Analysis completed: {analysis_config['analysis_completed']}")
                
                if analysis_config.get('coverage_quality'):
                    print(f"  Coverage quality: {analysis_config['coverage_quality']}")
                
                recommendations_count = analysis_config.get('accepted_recommendations', '0')
                print(f"  Accepted recommendations: {recommendations_count}")
            
            # Data integration settings
            if 'data_integration' in service_config:
                data_integration = service_config['data_integration']
                print(f"\nData Integration:")
                print(f"  Method: {data_integration['method']}")
                print(f"  Station data enabled: {data_integration['enable_station_data']}")
                
                distance_km = data_integration.get('local_station_distance_km', 'unknown')
                if distance_km != '999':
                    print(f"  Station distance: {distance_km} km")
            
            print(f"\n{CORE_ICONS['status']} Configuration complete - ready for installation!")

    def _handle_configuration_error(self, error, context="configuration"):
        """
        Handle configuration errors gracefully with helpful messaging
        Provides actionable error information to users
        """
        error_msg = str(error).lower()
        
        if "phase i" in error_msg or "marinedataservice" in error_msg:
            print(f"\n{CORE_ICONS['warning']} Phase I Dependency Error")
            print("The Phase I Marine Data Extension is required but not properly configured.")
            print("\nPlease ensure Phase I is installed and has marine stations selected:")
            print("1. Check if Phase I is installed: weectl extension list")
            print("2. Reconfigure Phase I if needed: sudo weectl extension reconfigure marine_data") 
            print("3. Restart WeeWX: sudo systemctl restart weewx")
            print("4. Then retry this installation")
            
        elif "grib" in error_msg or "eccodes" in error_msg or "pygrib" in error_msg:
            print(f"\n{CORE_ICONS['warning']} GRIB Processing Error")
            print("GRIB library installation or configuration failed.")
            print("\nYou can continue without WaveWatch III forecasts,")
            print("using only Phase I marine data for forecasting.")
            print("\nTo fix GRIB processing later:")
            print("  sudo apt-get install eccodes python3-eccodes")
            print("  # OR")
            print("  pip3 install eccodes-python")
            
        elif "permission" in error_msg or "sudo" in error_msg:
            print(f"\n{CORE_ICONS['warning']} Permission Error")
            print("Installation requires administrator privileges.")
            print("\nPlease run the installer with sudo:")
            print("  sudo weectl extension install weewx-surf-fishing-1.0.0-alpha.zip")
            
        elif "network" in error_msg or "connection" in error_msg:
            print(f"\n{CORE_ICONS['warning']} Network Error")
            print("Cannot connect to required services during installation.")
            print("\nPlease check your internet connection and try again.")
            print("Installation can continue with limited functionality.")
            
        else:
            print(f"\n{CORE_ICONS['warning']} Configuration Error")
            print(f"An error occurred during {context}: {error}")
            print("\nYou may need to:")
            print("1. Check system requirements")
            print("2. Verify WeeWX 5.1+ is installed")
            print("3. Ensure proper permissions")
            print("4. Check available disk space")
        
        print()

    def _validate_adaptive_configuration(self):
        """Validate that geographic regions and adaptive distances are properly configured"""
        
        print(f"\n{CORE_ICONS['status']} Validating adaptive distance configuration...")
        
        # Get configuration data from YAML (data-driven validation)
        bathymetry_data = self.yaml_data.get('bathymetry_data')
        if not bathymetry_data:
            print(f"  {CORE_ICONS['error']} No bathymetry_data section found in YAML")
            return False
            
        path_config = bathymetry_data.get('path_analysis')
        if not path_config:
            print(f"  {CORE_ICONS['error']} No path_analysis section found in YAML")
            return False
            
        regional_distances = path_config.get('regional_distances')
        if not regional_distances:
            print(f"  {CORE_ICONS['error']} No regional_distances found in YAML")
            return False
            
        geographic_regions = self.yaml_data.get('geographic_regions')
        if not geographic_regions:
            print(f"  {CORE_ICONS['error']} No geographic_regions section found in YAML")
            return False
        
        # Collect all defined region names from geographic boundaries (inline validation)
        all_defined_regions = set()
        for category_data in geographic_regions.values():
            if isinstance(category_data, dict):
                all_defined_regions.update(category_data.keys())
        
        if not all_defined_regions:
            print(f"  {CORE_ICONS['error']} No regions defined in geographic_regions")
            return False
        
        # Check for configuration mismatches (inline validation)
        distance_regions = set(regional_distances.keys())
        missing_boundaries = distance_regions - all_defined_regions
        missing_distances = all_defined_regions - distance_regions
        
        validation_passed = True
        
        if missing_boundaries:
            print(f"  {CORE_ICONS['error']} Regions with distances but no boundaries: {missing_boundaries}")
            validation_passed = False
        
        if missing_distances:
            print(f"  {CORE_ICONS['error']} Regions with boundaries but no distances: {missing_distances}")
            validation_passed = False
        
        if validation_passed:
            print(f"  {CORE_ICONS['status']} All regions properly configured")
            
        return validation_passed

    def _load_surf_depth_config(self):
        """
        Load surf break validation parameters from YAML bathymetry_data section
        """
        bathymetry_config = self.yaml_data.get('bathymetry_data', {})
        surf_validation = bathymetry_config.get('surf_break_validation', {})
        
        # Defaults if not in YAML (fallback)
        depth_range = surf_validation.get('optimal_depth_range', {})
        adjustment_config = surf_validation.get('depth_adjustment', {})
        
        return {
            'optimal_min': depth_range.get('min_meters', 1.5),
            'optimal_max': depth_range.get('max_meters', 6.0),
            'shallow_limit': adjustment_config.get('shallow_limit', 1.5),
            'deep_warning': adjustment_config.get('deep_warning', 6.0),
            'max_adjustment': adjustment_config.get('max_adjustment_meters', 250),
            'adjustment_step': adjustment_config.get('adjustment_step_meters', 50),
            'max_api_calls': adjustment_config.get('max_api_calls', 10)
        }

    def _try_coordinate_adjustment(self, lat, lon, beach_angle, current_depth, direction, spot_name, config):
        """
        Attempt coordinate adjustment along beach-perpendicular line using proper beach angle geometry
        """
        # Calculate correct bearing for movement direction
        if direction == 'seaward':
            movement_bearing = beach_angle  # Same direction as looking out to sea
        else:  # shoreward
            movement_bearing = (beach_angle + 180) % 360  # Opposite direction (toward land)
        
        step_size_degrees = config['adjustment_step'] / 111320  # Convert meters to degrees
        optimal_min, optimal_max = config['optimal_min'], config['optimal_max']
        
        api_calls_used = 0
        
        for step in range(1, int(config['max_adjustment'] / config['adjustment_step']) + 1):
            if api_calls_used >= config['max_api_calls']:
                print(f"  {CORE_ICONS['warning']} Reached API call limit for adjustments")
                break
                
            # Calculate new coordinates along movement bearing
            distance_degrees = step * step_size_degrees
            test_lat = lat + distance_degrees * math.cos(math.radians(movement_bearing))
            test_lon = lon + distance_degrees * math.sin(math.radians(movement_bearing)) / math.cos(math.radians(lat))
            
            # Test depth at new location
            self.progress.show_step_progress(f"Testing depth {step * config['adjustment_step']}m {direction}")
            is_water = self._validate_water_location(test_lat, test_lon)
            api_calls_used += 1
            
            if not is_water:
                # Hit land - stop adjustment in this direction
                print(f"  {CORE_ICONS['warning']} Reached land boundary during adjustment")
                break
                
            # Get depth at test location
            success, bathymetry_data, used_fallback = self.gebco_client.query_bathymetry_with_fallback(
                [(test_lat, test_lon)], self.progress)
            api_calls_used += 1
                
            if success and bathymetry_data:
                test_depth = bathymetry_data[0]
                
                # Check if we found suitable depth
                if optimal_min <= test_depth <= optimal_max:
                    self.progress.complete_step(f"Found suitable depth: {test_depth:.1f}m")
                    return test_lat, test_lon, test_depth
                    
                # For shallow adjustment, any deeper water helps
                if direction == 'seaward' and test_depth > current_depth:
                    continue
                    
                # For deep adjustment, any shallower water helps  
                if direction == 'shoreward' and test_depth < current_depth:
                    continue
        
        return None, None, None

    def _handle_adjustment_failure(self, lat, lon, current_depth, spot_name, depth_issue):
        """
        Handle cases where coordinate adjustment failed to find suitable depth with user override options
        """
        print(f"  {CORE_ICONS['warning']} Could not find suitable surf break depth within 250m")
        print(f"  Current location: {current_depth:.1f}m depth")
        print(f"  This may indicate unusual bathymetry that could greatly impact forecast accuracy.")
        print()
        print("  Options:")
        print("  1. Keep current coordinates (may affect forecast accuracy)")
        print("  2. Enter new surf break coordinates")
        
        while True:
            choice = input("  Choice (1-2): ").strip()
            if choice == '1':
                print(f"  {CORE_ICONS['warning']} Using coordinates with {depth_issue} depth - forecast accuracy may be affected")
                return lat, lon
            elif choice == '2':
                print(f"  Please enter new coordinates for {spot_name} surf break")
                return self._get_coordinates_for_water_location("surf break")
            else:
                print(f"  {CORE_ICONS['warning']} Please enter 1 or 2")

    def _find_weewx_config_path(self):
        """Find weewx.conf path using WeeWX 5.1 standard locations"""
        possible_paths = [
            '/etc/weewx/weewx.conf',
            '/home/weewx/weewx.conf',
            '~/weewx-data/weewx.conf',
            '/opt/weewx/weewx.conf'
        ]
        
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        return None

    def _check_existing_spots_in_conf(self):
        """Check for existing spots in weewx.conf and return counts"""
        try:
            config_path = self._find_weewx_config_path()
            if not config_path or not os.path.exists(config_path):
                return {'surf_count': 0, 'fishing_count': 0}
            
            config = configobj.ConfigObj(config_path, interpolation=False)
            service_config = config.get('SurfFishingService', {})
            
            surf_count = len(service_config.get('surf_spots', {}))
            fishing_count = len(service_config.get('fishing_spots', {}))
            
            return {'surf_count': surf_count, 'fishing_count': fishing_count}
        except Exception:
            return {'surf_count': 0, 'fishing_count': 0}              

    def _configure_locations_with_cursor_ui(self, forecast_types):
        """Configure locations using your integrated cursor UI class"""
        
        print(f"\n{CORE_ICONS['selection']} Enhanced Point Management System")
        print("Using interactive cursor interface for surf and fishing point management")
        print()
        
        # Initialize your cursor UI class (assuming it's named SurfFishingPointManager)
        point_manager = SurfFishingPointManager(self.yaml_data, self.gebco_client)
        management_result = point_manager.run_management_interface()
        
        print(f"\n{CORE_ICONS['status']} Point management completed")
        if management_result.get('changes_made', False):
            print(f"  • Changes saved to weewx.conf")
        if management_result.get('config_loaded', False):
            print(f"  • Loaded existing configuration successfully")
        
        return {
            'surf_spots': management_result.get('surf_spots', []),
            'fishing_spots': management_result.get('fishing_spots', [])
        }
        
    def _configure_locations_manually(self, forecast_types):
        """Configure locations using existing manual methods (preserved functionality)"""
        
        print(f"\n{CORE_ICONS['selection']} Manual Point Configuration")
        print("Using traditional manual entry interface")
        print()
        
        selected_locations = {}
        
        if 'surf' in forecast_types:
            selected_locations['surf_spots'] = self._configure_surf_spots()
        
        if 'fishing' in forecast_types:
            selected_locations['fishing_spots'] = self._configure_fishing_spots()
        
        return selected_locations

                
class PhaseIAnalyzer:
    """
    Analyze existing Phase I station configuration from CONF metadata
    Provides zero-API station analysis for Phase II optimization
    """
    
    def __init__(self, config_dict):
        """
        Initialize analyzer with WeeWX configuration dictionary
        """
        self.config_dict = config_dict
        self.marine_config = config_dict.get('MarineDataService', {})
        self.station_metadata = self.marine_config.get('station_metadata', {})
        
    def get_available_stations(self):
        """
        Extract all Phase I selected stations with metadata from CONF
        Returns comprehensive station data without any API calls
        FIXED: Corrected capability string matching for actual CONF data
        """
        stations = {
            'coops_stations': [],
            'ndbc_stations': [],
            'total_count': 0
        }
        
        # Process CO-OPS stations from Phase I metadata
        coops_metadata = self.station_metadata.get('coops_stations', {})
        for station_id, metadata in coops_metadata.items():
            # FIXED: Proper capability string parsing for your actual CONF data
            capabilities_str = metadata.get('capabilities', '')
            
            # Your CONF has "tide_predictions" - handle both single strings and comma-separated
            capabilities_list = []
            if isinstance(capabilities_str, str):
                if ',' in capabilities_str:
                    capabilities_list = [cap.strip() for cap in capabilities_str.split(',')]
                else:
                    capabilities_list = [capabilities_str.strip()]
            
            # FIXED: Accurate capability detection for CO-OPS stations
            has_tide_data = any(cap in ['water_level', 'predictions', 'tide_predictions'] for cap in capabilities_list)
            has_water_temp = any(cap in ['water_temperature', 'water_temp'] for cap in capabilities_list)
            
            station_info = {
                'station_id': station_id,
                'name': metadata.get('name', f'Station {station_id}'),
                'latitude': float(metadata.get('latitude', 0)),
                'longitude': float(metadata.get('longitude', 0)),
                'distance_miles': float(metadata.get('distance_miles', 0)),
                'capabilities': capabilities_list,
                'station_type': metadata.get('station_type', 'coops'),
                'has_tide_data': has_tide_data,
                'has_water_temp': has_water_temp
            }
            stations['coops_stations'].append(station_info)
        
        # Process NDBC stations from Phase I metadata  
        ndbc_metadata = self.station_metadata.get('ndbc_stations', {})
        for station_id, metadata in ndbc_metadata.items():
            # FIXED: Direct boolean reading from CONF metadata
            wave_capability = metadata.get('wave_capability', 'false')
            atmospheric_capability = metadata.get('atmospheric_capability', 'false')
            
            # Handle both string and boolean values
            if isinstance(wave_capability, str):
                has_wave_data = wave_capability.lower() == 'true'
            else:
                has_wave_data = bool(wave_capability)
                
            if isinstance(atmospheric_capability, str):
                has_atmospheric_data = atmospheric_capability.lower() == 'true'
            else:
                has_atmospheric_data = bool(atmospheric_capability)
            
            capabilities = {
                'wave_data': has_wave_data,
                'atmospheric_data': has_atmospheric_data
            }
            
            station_info = {
                'station_id': station_id,
                'name': metadata.get('name', f'Buoy {station_id}'),
                'latitude': float(metadata.get('latitude', 0)),
                'longitude': float(metadata.get('longitude', 0)),
                'distance_miles': float(metadata.get('distance_miles', 0)),
                'capabilities': capabilities,
                'station_type': metadata.get('station_type', 'ndbc'),
                'has_wave_data': has_wave_data,
                'has_atmospheric_data': has_atmospheric_data
            }
            stations['ndbc_stations'].append(station_info)
        
        stations['total_count'] = len(stations['coops_stations']) + len(stations['ndbc_stations'])
        return stations

    def analyze_phase_i_coverage(self):
        """
        Analyze Phase I station coverage for basic metrics
        Returns coverage summary for user information
        """
        stations = self.get_available_stations()
        
        analysis = {
            'total_stations': stations['total_count'],
            'coops_count': len(stations['coops_stations']),
            'ndbc_count': len(stations['ndbc_stations']),
            'coverage_types': set(),
            'distance_summary': {
                'closest_station': None,
                'average_distance': 0,
                'farthest_station': None
            }
        }
        
        all_stations = stations['coops_stations'] + stations['ndbc_stations']
        
        if all_stations:
            # Calculate distance statistics
            distances = [s['distance_miles'] for s in all_stations]
            analysis['distance_summary']['average_distance'] = sum(distances) / len(distances)
            
            closest = min(all_stations, key=lambda s: s['distance_miles'])
            analysis['distance_summary']['closest_station'] = {
                'name': closest['name'],
                'distance': closest['distance_miles'],
                'type': closest['station_type']
            }
            
            farthest = max(all_stations, key=lambda s: s['distance_miles'])
            analysis['distance_summary']['farthest_station'] = {
                'name': farthest['name'], 
                'distance': farthest['distance_miles'],
                'type': farthest['station_type']
            }
            
            # Identify coverage types
            for station in stations['coops_stations']:
                if station['has_tide_data']:
                    analysis['coverage_types'].add('tide_data')
                if station['has_water_temp']:
                    analysis['coverage_types'].add('water_temperature')
            
            for station in stations['ndbc_stations']:
                if station['has_wave_data']:
                    analysis['coverage_types'].add('wave_data')
                if station['has_atmospheric_data']:
                    analysis['coverage_types'].add('atmospheric_data')
        
        return analysis
    

class StationQualityAnalyzer:
    """
    Analyze station data quality for user locations using research-based thresholds
    Implements coastal oceanography research for distance-based quality scoring
    """
    
    def __init__(self, yaml_data):
        """
        Initialize with quality thresholds from YAML configuration
        """
        # Load research-based quality thresholds from YAML
        quality_config = yaml_data.get('station_quality_thresholds', {})
        
        self.wave_thresholds = quality_config.get('wave_data', {
            'excellent_distance_km': 40,  # 25 miles
            'good_distance_km': 80,       # 50 miles  
            'fair_distance_km': 160,      # 100 miles
            'minimum_quality': 0.6
        })
        
        self.atmospheric_thresholds = quality_config.get('atmospheric_data', {
            'excellent_distance_km': 80,   # 50 miles
            'good_distance_km': 160,       # 100 miles
            'fair_distance_km': 320,       # 200 miles
            'minimum_quality': 0.5
        })
        
        self.tide_thresholds = quality_config.get('tide_data', {
            'excellent_distance_km': 80,   # 50 miles
            'good_distance_km': 160,       # 100 miles  
            'fair_distance_km': 240,       # 150 miles
            'minimum_quality': 0.7
        })
    
    def calculate_wave_quality_score(self, distance_miles):
        """
        Calculate wave data quality score based on research-validated distance thresholds
        Returns quality score (0.0-1.0) and quality level
        """
        distance_km = distance_miles * 1.60934
        
        if distance_km <= self.wave_thresholds['excellent_distance_km']:
            return 1.0, 'excellent'
        elif distance_km <= self.wave_thresholds['good_distance_km']:
            return 0.8, 'good'
        elif distance_km <= self.wave_thresholds['fair_distance_km']:
            return 0.6, 'fair'
        else:
            # Linear decay beyond fair threshold
            max_distance = self.wave_thresholds['fair_distance_km'] * 2
            if distance_km >= max_distance:
                return 0.1, 'poor'
            decay_factor = (max_distance - distance_km) / max_distance
            return max(0.1, decay_factor * 0.5), 'poor'
    
    def calculate_atmospheric_quality_score(self, distance_miles):
        """
        Calculate atmospheric data quality score based on research thresholds
        Returns quality score (0.0-1.0) and quality level
        """
        distance_km = distance_miles * 1.60934
        
        if distance_km <= self.atmospheric_thresholds['excellent_distance_km']:
            return 1.0, 'excellent'
        elif distance_km <= self.atmospheric_thresholds['good_distance_km']:
            return 0.8, 'good'
        elif distance_km <= self.atmospheric_thresholds['fair_distance_km']:
            return 0.6, 'fair'
        else:
            max_distance = self.atmospheric_thresholds['fair_distance_km'] * 2
            if distance_km >= max_distance:
                return 0.1, 'poor'
            decay_factor = (max_distance - distance_km) / max_distance
            return max(0.1, decay_factor * 0.5), 'poor'
    
    def calculate_tide_quality_score(self, distance_miles):
        """
        Calculate tide data quality score based on coastal research
        Returns quality score (0.0-1.0) and quality level
        """
        distance_km = distance_miles * 1.60934
        
        if distance_km <= self.tide_thresholds['excellent_distance_km']:
            return 1.0, 'excellent'
        elif distance_km <= self.tide_thresholds['good_distance_km']:
            return 0.8, 'good'
        elif distance_km <= self.tide_thresholds['fair_distance_km']:
            return 0.6, 'fair'
        else:
            max_distance = self.tide_thresholds['fair_distance_km'] * 2
            if distance_km >= max_distance:
                return 0.1, 'poor'
            decay_factor = (max_distance - distance_km) / max_distance
            return max(0.1, decay_factor * 0.5), 'poor'

    def analyze_location_coverage(self, location, available_stations):
        """
        Analyze data coverage quality for specific user location
        Returns comprehensive coverage analysis with quality scores
        """
        location_lat = location['latitude']
        location_lon = location['longitude']
        location_name = location['name']
        
        analysis = {
            'location': location_name,
            'coordinates': (location_lat, location_lon),
            'wave_sources': [],
            'atmospheric_sources': [],
            'tide_sources': [],
            'quality_summary': {
                'wave_quality': 0.0,
                'atmospheric_quality': 0.0,
                'tide_quality': 0.0,
                'overall_quality': 0.0
            },
            'recommendations': []
        }
        
        # Analyze NDBC stations for wave and atmospheric data
        for station in available_stations['ndbc_stations']:
            station_distance = self._calculate_distance(
                location_lat, location_lon,
                station['latitude'], station['longitude']
            )
            
            if station['has_wave_data']:
                wave_score, wave_level = self.calculate_wave_quality_score(station_distance)
                analysis['wave_sources'].append({
                    'station_id': station['station_id'],
                    'name': station['name'],
                    'distance_miles': station_distance,
                    'quality_score': wave_score,
                    'quality_level': wave_level
                })
            
            if station['has_atmospheric_data']:
                atmo_score, atmo_level = self.calculate_atmospheric_quality_score(station_distance)
                analysis['atmospheric_sources'].append({
                    'station_id': station['station_id'],
                    'name': station['name'],
                    'distance_miles': station_distance,
                    'quality_score': atmo_score,
                    'quality_level': atmo_level
                })
        
        # Analyze CO-OPS stations for tide data
        for station in available_stations['coops_stations']:
            if station['has_tide_data']:
                station_distance = self._calculate_distance(
                    location_lat, location_lon,
                    station['latitude'], station['longitude']
                )
                
                tide_score, tide_level = self.calculate_tide_quality_score(station_distance)
                analysis['tide_sources'].append({
                    'station_id': station['station_id'],
                    'name': station['name'],
                    'distance_miles': station_distance,
                    'quality_score': tide_score,
                    'quality_level': tide_level
                })
        
        # Calculate best quality scores for each data type
        if analysis['wave_sources']:
            analysis['quality_summary']['wave_quality'] = max(s['quality_score'] for s in analysis['wave_sources'])
        
        if analysis['atmospheric_sources']:
            analysis['quality_summary']['atmospheric_quality'] = max(s['quality_score'] for s in analysis['atmospheric_sources'])
        
        if analysis['tide_sources']:
            analysis['quality_summary']['tide_quality'] = max(s['quality_score'] for s in analysis['tide_sources'])
        
        # Calculate overall quality (weighted average)
        weights = {'wave': 0.4, 'atmospheric': 0.4, 'tide': 0.2}
        overall_quality = (
            analysis['quality_summary']['wave_quality'] * weights['wave'] +
            analysis['quality_summary']['atmospheric_quality'] * weights['atmospheric'] +
            analysis['quality_summary']['tide_quality'] * weights['tide']
        )
        analysis['quality_summary']['overall_quality'] = overall_quality
        
        return analysis
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using haversine formula
        Returns distance in miles
        """
        import math
        R = 3959  # Earth radius in miles
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2) * math.sin(delta_lat/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon/2) * math.sin(delta_lon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    

class MarineStationRecommendationEngine:
    """
    Generate station recommendations for optimizing surf and fishing forecasts
    Provides actionable suggestions for improving data coverage
    """
    
    def __init__(self, phase_i_analyzer, quality_analyzer):
        """
        Initialize recommendation engine with analyzers
        """
        self.phase_i_analyzer = phase_i_analyzer
        self.quality_analyzer = quality_analyzer
    
    def analyze_multi_location_optimization(self, user_locations):
        """
        Analyze all user locations and provide comprehensive optimization recommendations
        Returns prioritized recommendations for improving overall data coverage
        """
        available_stations = self.phase_i_analyzer.get_available_stations()
        
        # Analyze coverage for each location
        location_analyses = []
        for location in user_locations:
            analysis = self.quality_analyzer.analyze_location_coverage(location, available_stations)
            location_analyses.append(analysis)
        
        # Generate comprehensive recommendations
        recommendations = {
            'priority_recommendations': [],
            'optimization_summary': {},
            'current_coverage': {},
            'improvement_potential': {}
        }
        
        # Analyze overall coverage quality
        overall_wave_quality = sum(a['quality_summary']['wave_quality'] for a in location_analyses) / len(location_analyses) if location_analyses else 0
        overall_atmospheric_quality = sum(a['quality_summary']['atmospheric_quality'] for a in location_analyses) / len(location_analyses) if location_analyses else 0
        overall_tide_quality = sum(a['quality_summary']['tide_quality'] for a in location_analyses) / len(location_analyses) if location_analyses else 0
        
        recommendations['current_coverage'] = {
            'wave_quality': overall_wave_quality,
            'atmospheric_quality': overall_atmospheric_quality, 
            'tide_quality': overall_tide_quality,
            'total_locations': len(location_analyses)
        }
        
        # Generate priority recommendations based on quality gaps
        if overall_wave_quality < 0.6:
            recommendations['priority_recommendations'].append({
                'type': 'critical',
                'category': 'wave_data',
                'message': f'Wave data quality is below recommended threshold ({overall_wave_quality:.1f}/1.0)',
                'impact': 'Surf forecasts may be less accurate due to distant wave buoys',
                'suggestion': 'Consider adding closer NDBC wave buoys to Phase I configuration'
            })
        
        if overall_atmospheric_quality < 0.5:
            recommendations['priority_recommendations'].append({
                'type': 'critical',
                'category': 'atmospheric_data',
                'message': f'Atmospheric data quality is below threshold ({overall_atmospheric_quality:.1f}/1.0)',
                'impact': 'Wind and pressure forecasts may lack local precision',
                'suggestion': 'Add closer NDBC atmospheric buoys or enable WeeWX station integration'
            })
        
        if overall_tide_quality < 0.7:
            recommendations['priority_recommendations'].append({
                'type': 'important',
                'category': 'tide_data',
                'message': f'Tide data quality could be improved ({overall_tide_quality:.1f}/1.0)',
                'impact': 'Fishing forecasts may be less precise for tide-dependent species',
                'suggestion': 'Consider adding backup CO-OPS tide stations'
            })
        
        # Check for locations with particularly poor coverage
        for analysis in location_analyses:
            location_quality = analysis['quality_summary']['overall_quality']
            if location_quality < 0.4:
                recommendations['priority_recommendations'].append({
                    'type': 'location_specific',
                    'category': 'poor_coverage',
                    'message': f'Location "{analysis["location"]}" has poor data coverage ({location_quality:.1f}/1.0)',
                    'impact': 'Forecasts for this location may be unreliable',
                    'suggestion': f'Consider relocating closer to existing stations or adding stations near {analysis["location"]}'
                })
        
        return recommendations
    
    def display_recommendations_interactive(self, recommendations):
        """
        Display recommendations to user and get their choices
        Returns list of accepted recommendations for implementation
        """
        if not recommendations['priority_recommendations']:
            print(f"\n{CORE_ICONS['status']} Station Coverage Analysis")
            print("Your Phase I station configuration provides good coverage for all locations.")
            print("No immediate improvements needed.")
            return []
        
        print(f"\n{CORE_ICONS['selection']} Station Coverage Recommendations")
        print("Based on your surf/fishing locations, we recommend these improvements:")
        print()
        
        accepted_recommendations = []
        for i, rec in enumerate(recommendations['priority_recommendations'], 1):
            icon = CORE_ICONS['warning'] if rec['type'] == 'critical' else CORE_ICONS['navigation']
            print(f"{icon} {i}. {rec['message']}")
            print(f"   Impact: {rec['impact']}")
            print(f"   Suggestion: {rec['suggestion']}")
            
            while True:
                choice = input(f"   Apply this recommendation? (y/n): ").strip().lower()
                if choice in ['y', 'yes']:
                    accepted_recommendations.append(rec)
                    print(f"   {CORE_ICONS['status']} Will implement")
                    break
                elif choice in ['n', 'no']:
                    print(f"   Skipped")
                    break
                else:
                    print(f"   {CORE_ICONS['warning']} Please enter y or n")
            print()
        
        return accepted_recommendations


class SurfFishingInstaller(ExtensionInstaller):
    """
    WeeWX Surf & Fishing Forecast Extension Installer
    Follows WeeWX 5.1 ExtensionInstaller patterns exactly
    """
    
    def __init__(self):
        super(SurfFishingInstaller, self).__init__(
            version="1.0.0-alpha",
            name="surf_fishing_forecasts",
            description="Local surf and fishing forecast system for WeeWX",
            author="Shane Burkhardt",
            author_email="your-email@domain.com",
            
            # CRITICAL: Use LIST format for services
            data_services=['user.surf_fishing.SurfFishingService'],
            
            files=[
                ('bin/user', ['bin/user/surf_fishing.py']),
                ('bin/user', ['bin/user/surf_fishing_fields.yaml'])
            ],
            
            config={
                'SurfFishingService': {
                    'enable': 'true',
                    'forecast_interval': '21600',  # 6 hours in seconds
                    'log_success': 'false',
                    'log_errors': 'true'
                }
            }
        )
        
        # Load YAML configuration data for installer use
        self.yaml_data = self._load_yaml_data()
    
    def _load_yaml_data(self):
        """Load 4-section reorganized YAML configuration data for installation"""
        yaml_file = 'bin/user/surf_fishing_fields.yaml'
        
        try:
            if not os.path.exists(yaml_file):
                log.error(f"YAML configuration file not found: {yaml_file}")
                print(f"\n{CORE_ICONS['warning']} ERROR: Required YAML file not found: {yaml_file}")
                print("Installation cannot proceed without configuration file.")
                sys.exit(1)
            
            with open(yaml_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # CRITICAL: Validate 4-section structure
            required_sections = ['noaa_gfs_wave', 'bathymetry_data', 'fish_categories', 'scoring_criteria']
            missing_sections = [section for section in required_sections if section not in yaml_data]
            
            if missing_sections:
                error_msg = f"YAML missing required sections: {', '.join(missing_sections)}"
                print(f"\n{CORE_ICONS['warning']} ERROR: {error_msg}")
                print("The surf_fishing_fields.yaml file must contain all 4 sections:")
                for section in required_sections:
                    status = CORE_ICONS['status'] if section in yaml_data else CORE_ICONS['warning']
                    print(f"  {status} {section}")
                sys.exit(1)
            
            # Validate field mappings within noaa_gfs_wave section
            gfs_wave_section = yaml_data.get('noaa_gfs_wave', {})
            field_mappings = gfs_wave_section.get('field_mappings', {})
            
            if not field_mappings:
                print(f"\n{CORE_ICONS['warning']} ERROR: No field_mappings found in noaa_gfs_wave section")
                sys.exit(1)
            
            field_count = len(field_mappings)
            print(f"{CORE_ICONS['status']} Loaded 4-section YAML with {field_count} GFS Wave fields")
            
            return yaml_data
            
        except yaml.YAMLError as e:
            print(f"\n{CORE_ICONS['warning']} ERROR: Invalid YAML format: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"\n{CORE_ICONS['warning']} ERROR: Could not load YAML configuration: {e}")
            sys.exit(1)
    
    def configure(self, engine):
        """
        Called by WeeWX during installation
        Returns True for success, False for failure
        """
        
        try:
            print("\n" + "="*80)
            print("SURF & FISHING FORECAST EXTENSION INSTALLATION")
            print("="*80)
            print("Phase II: Local Surf & Fishing Forecast System")
            print("Magic Animal: Seahorse 🐟")
            print("-" * 80)
            
            # Interactive configuration
            configurator = SurfFishingConfigurator(engine.config_dict, self.yaml_data)
            config_dict, selected_locations = configurator.run_interactive_setup()
            
            # Extend database schema
            self._extend_database_schema(engine.config_dict, selected_locations)
            
            # Update engine configuration
            engine.config_dict.update(config_dict)
            
            print("\n" + "="*80)
            print("INSTALLATION COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"{CORE_ICONS['status']} Files installed")
            print(f"{CORE_ICONS['status']} Service registered automatically")
            print(f"{CORE_ICONS['status']} Interactive configuration completed")
            print(f"{CORE_ICONS['status']} Database schema extended")
            print("-" * 80)
            print("NEXT STEPS:")
            print("1. Restart WeeWX to activate the extension:")
            print("   sudo systemctl restart weewx")
            print("2. Check WeeWX logs for successful startup:")
            print("   sudo tail -f /var/log/syslog | grep weewx")
            print("3. Forecasts will be generated every 6 hours")
            print("="*80)
            
            return True
            
        except Exception as e:
            print(f"\nInstallation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extend_database_schema(self, config_dict, selected_locations):
        """Extract field definitions from 4-section YAML and create database fields"""
        
        progress = InstallationProgressManager()
        
        try:
            progress.show_step_progress("Analyzing Database Requirements")
            
            # Extract field mappings from noaa_gfs_wave section
            gfs_wave_section = self.yaml_data.get('noaa_gfs_wave', {})
            field_mappings = gfs_wave_section.get('field_mappings', {})
            
            if not field_mappings:
                raise Exception(
                    f"{CORE_ICONS['warning']} No field_mappings found in noaa_gfs_wave YAML section. "
                    "Cannot create database fields without field definitions."
                )
            
            progress.complete_step("Analyzing Database Requirements")
            progress.show_step_progress("Creating Database Fields")
            
            # Build field definitions for database creation
            required_fields = {}
            
            # Add forecast data tables (preserve existing logic)
            forecast_tables = [
                'marine_forecast_surf_data',
                'marine_forecast_fishing_data'
            ]
            
            for table_name in forecast_tables:
                try:
                    # Use WeeWX 5.1 database manager patterns
                    db_manager = weewx.manager.open_manager_with_config(config_dict, 'wx_binding')
                    
                    # Check if table exists
                    if not self._table_exists(db_manager, table_name):
                        # Create table with basic structure
                        self._create_forecast_table(db_manager, table_name)
                        print(f"    {CORE_ICONS['status']} Created table: {table_name}")
                    else:
                        print(f"    {CORE_ICONS['status']} Table exists: {table_name}")
                    
                    db_manager.close()
                    
                except Exception as e:
                    progress.show_error("Creating Database Fields", f"Table {table_name}: {e}")
                    continue
            
            # Extract GFS Wave fields from YAML field_mappings
            gfs_wave_fields = {}
            for field_name, field_config in field_mappings.items():
                database_field = field_config.get('database_field', field_name)
                database_type = field_config.get('database_type', 'REAL')
                gfs_wave_fields[database_field] = database_type
            
            # Add GFS Wave fields to archive table if needed
            if gfs_wave_fields:
                try:
                    db_manager = weewx.manager.open_manager_with_config(config_dict, 'wx_binding')
                    missing_fields = self._check_missing_fields(db_manager, gfs_wave_fields)
                    
                    if missing_fields:
                        self._add_missing_fields(db_manager, missing_fields)
                        print(f"    {CORE_ICONS['status']} Added {len(missing_fields)} GFS Wave fields")
                    else:
                        print(f"    {CORE_ICONS['status']} All GFS Wave fields exist")
                    
                    db_manager.close()
                    
                except Exception as e:
                    progress.show_error("Creating Database Fields", f"GFS Wave fields: {e}")
            
            progress.complete_step("Creating Database Fields")
            print(f"  {CORE_ICONS['status']} Database schema updated successfully")
            
        except Exception as e:
            progress.show_error("Database Schema Extension", str(e))
            raise

    def _table_exists(self, db_manager, table_name):
        """Check if database table exists"""
        try:
            result = db_manager.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
                (table_name,)
            )
            return result.fetchone() is not None
        except Exception:
            return False

    def _create_forecast_table(self, db_manager, table_name):
        """
        Create forecast data table using YAML field mappings + hardcoded service fields
        Uses correct WeeWX 5.1 database patterns from original method
        NO FALLBACK - fail if YAML field mappings missing
        """
        
        # Extract API field definitions from YAML field mappings
        gfs_wave_section = self.yaml_data.get('noaa_gfs_wave', {})
        field_mappings = gfs_wave_section.get('field_mappings', {})
        
        if not field_mappings:
            raise Exception(f"Cannot create table '{table_name}' - no field_mappings found in YAML noaa_gfs_wave section")
        
        # Build column definitions
        column_sql = []
        
        # HARDCODED SERVICE FIELDS (always required)
        if table_name == 'marine_forecast_surf_data':
            # WeeWX standard fields
            column_sql.append("dateTime INTEGER NOT NULL")
            column_sql.append("usUnits INTEGER NOT NULL")
            
            # Service-specific key fields
            column_sql.append("spot_id VARCHAR(50) NOT NULL")
            column_sql.append("forecast_time INTEGER NOT NULL")
            column_sql.append("generated_time INTEGER NOT NULL")
            
            # Service-calculated fields
            column_sql.append("quality_rating INTEGER")
            column_sql.append("confidence REAL")
            column_sql.append("conditions_text TEXT")
            column_sql.append("wind_condition TEXT")
            column_sql.append("tide_height REAL")
            column_sql.append("tide_stage TEXT")
            column_sql.append("wave_height_min REAL")
            column_sql.append("wave_height_max REAL")
            column_sql.append("wave_height_range TEXT")
            column_sql.append("quality_stars INTEGER")
            column_sql.append("quality_text TEXT")
            column_sql.append("conditions_description TEXT")
            column_sql.append("swell_dominance TEXT")
            
            # Primary key
            primary_key = "PRIMARY KEY (dateTime, spot_id, forecast_time)"
            
        elif table_name == 'marine_forecast_fishing_data':
            # WeeWX standard fields
            column_sql.append("dateTime INTEGER NOT NULL")
            column_sql.append("usUnits INTEGER NOT NULL")
            
            # Service-specific key fields
            column_sql.append("spot_id VARCHAR(50) NOT NULL")
            column_sql.append("forecast_date INTEGER NOT NULL")
            column_sql.append("period_name VARCHAR(50) NOT NULL")  # Fixed: VARCHAR with length
            column_sql.append("period_start_hour INTEGER")
            column_sql.append("period_end_hour INTEGER")
            column_sql.append("generated_time INTEGER NOT NULL")
            
            # Service-calculated fields
            column_sql.append("pressure_trend TEXT")
            column_sql.append("tide_movement TEXT")
            column_sql.append("species_activity TEXT")
            column_sql.append("activity_rating INTEGER")
            column_sql.append("conditions_text TEXT")
            column_sql.append("best_species TEXT")
            
            # Primary key
            primary_key = "PRIMARY KEY (dateTime, spot_id, forecast_date, period_name)"
            
        else:
            raise Exception(f"Unknown forecast table: {table_name}")
        
        # ADD API FIELDS from YAML field mappings
        if table_name == 'marine_forecast_surf_data':
            for field_name, field_config in field_mappings.items():
                database_field = field_config.get('database_field', field_name)
                database_type = field_config.get('database_type', 'REAL')
                
                # Add API field to surf table only
                column_sql.append(f"{database_field} {database_type}")
        # Fishing table gets NO API fields - it only uses calculated/derived fields

        # Add primary key constraint
        column_sql.append(primary_key)
        
        # Build CREATE TABLE statement
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {',\n    '.join(column_sql)}\n)"
        
        # ✅ PRESERVE ORIGINAL WeeWX 5.1 PATTERN EXACTLY
        db_manager.connection.execute(sql)
        db_manager.connection.commit()
        
        print(f"    {CORE_ICONS['status']} Created table {table_name} with {len(field_mappings)} API fields + service fields")

    def _check_missing_fields(self, db_manager, required_fields):
        """Check which fields are missing from archive table"""
        
        try:
            # Get existing columns
            result = db_manager.connection.execute("PRAGMA table_info(archive)")
            existing_columns = [row[1] for row in result.fetchall()]
            
            # Find missing fields
            missing_fields = {}
            for field_name, field_type in required_fields.items():
                if field_name not in existing_columns:
                    missing_fields[field_name] = field_type
            
            return missing_fields
            
        except Exception as e:
            log.error(f"Error checking missing fields: {e}")
            return {}

    def _add_missing_fields(self, db_manager, missing_fields):
        """Add missing fields to archive table using WeeWX 5.1 patterns"""
        
        for field_name, field_type in missing_fields.items():
            try:
                # Use ALTER TABLE for adding fields
                sql = f"ALTER TABLE archive ADD COLUMN {field_name} {field_type}"
                db_manager.connection.execute(sql)
                db_manager.connection.commit()
                print(f"      {CORE_ICONS['status']} Added field: {field_name} ({field_type})")
                
            except Exception as e:
                print(f"      {CORE_ICONS['warning']} Could not add field {field_name}: {e}")
                continue

    def _add_user_locations_to_conf(self, config_dict, selected_locations):
        """
        Add user-configured locations to WeeWX configuration (CONF-based approach)
        Uses data-driven structure from YAML configuration
        """
        
        # Ensure SurfFishingService section exists
        if 'SurfFishingService' not in config_dict:
            config_dict['SurfFishingService'] = {}
        
        service_config = config_dict['SurfFishingService']
        
        # Get location configuration structure from YAML
        location_config_template = self.yaml_data.get('location_config', {})
        
        # Add surf spots to CONF using data-driven structure
        if 'surf_spots' in selected_locations and selected_locations['surf_spots']:
            service_config['surf_spots'] = {}
            
            for i, spot in enumerate(selected_locations['surf_spots']):
                spot_key = f'spot_{i+1}'
                
                # Use YAML template structure if available
                spot_config = self._build_spot_config_from_yaml(spot, 'surf', location_config_template)
                service_config['surf_spots'][spot_key] = spot_config
        
        # Add fishing spots to CONF using data-driven structure  
        if 'fishing_spots' in selected_locations and selected_locations['fishing_spots']:
            service_config['fishing_spots'] = {}
            
            for i, spot in enumerate(selected_locations['fishing_spots']):
                spot_key = f'spot_{i+1}'
                
                # Use YAML template structure if available
                spot_config = self._build_spot_config_from_yaml(spot, 'fishing', location_config_template)
                service_config['fishing_spots'][spot_key] = spot_config
        
        # Add location metadata using YAML configuration
        location_metadata = self._get_location_metadata_from_yaml(selected_locations)
        if location_metadata:
            service_config['location_metadata'] = location_metadata
        
        surf_count = len(selected_locations.get('surf_spots', []))
        fishing_count = len(selected_locations.get('fishing_spots', []))
        
        print(f"  {CORE_ICONS['status']} Added {surf_count} surf spots and {fishing_count} fishing spots to configuration")

    def _build_spot_config_from_yaml(self, spot_data, spot_type, location_config_template):
        """
        Build spot configuration using YAML template structure
        Data-driven approach ensures consistency with YAML definitions
        """
        
        # Get template structure from YAML
        template = location_config_template.get(f'{spot_type}_spot_template', {})
        
        # Start with required fields (always strings for CONF)
        spot_config = {
            'name': spot_data['name'],
            'latitude': str(spot_data['latitude']),
            'longitude': str(spot_data['longitude']),
            'type': spot_type,
            'active': 'true'
        }
        
        # Add type-specific fields based on YAML template
        if spot_type == 'surf':
            # Add surf-specific fields from YAML template
            surf_fields = template.get('fields', {})
            if 'bottom_type' in surf_fields:
                spot_config['bottom_type'] = spot_data.get('bottom_type', surf_fields['bottom_type'].get('default', 'sand'))
            if 'exposure' in surf_fields:
                spot_config['exposure'] = spot_data.get('exposure', surf_fields['exposure'].get('default', 'exposed'))
                
        elif spot_type == 'fishing':
            # Add fishing-specific fields from YAML template
            fishing_fields = template.get('fields', {})
            if 'location_type' in fishing_fields:
                spot_config['location_type'] = spot_data.get('location_type', fishing_fields['location_type'].get('default', 'shore'))
            if 'target_category' in fishing_fields:
                spot_config['target_category'] = spot_data.get('target_category', fishing_fields['target_category'].get('default', 'mixed_bag'))
        
        # Add any additional fields defined in YAML template
        additional_fields = template.get('additional_fields', {})
        for field_name, field_config in additional_fields.items():
            if field_name not in spot_config:
                default_value = field_config.get('default', '')
                spot_config[field_name] = str(spot_data.get(field_name, default_value))
        
        return spot_config

    def _get_location_metadata_from_yaml(self, selected_locations):
        """
        Extract location metadata using YAML configuration
        Data-driven approach for consistent metadata handling
        """
        
        metadata_template = self.yaml_data.get('location_metadata', {})
        
        if not metadata_template:
            return None
        
        metadata = {}
        
        # Add timestamp
        if 'include_timestamp' in metadata_template and metadata_template['include_timestamp']:
            metadata['configured_date'] = str(int(time.time()))
        
        # Add location counts
        if 'include_counts' in metadata_template and metadata_template['include_counts']:
            metadata['total_surf_spots'] = str(len(selected_locations.get('surf_spots', [])))
            metadata['total_fishing_spots'] = str(len(selected_locations.get('fishing_spots', [])))
            metadata['total_locations'] = str(
                len(selected_locations.get('surf_spots', [])) + 
                len(selected_locations.get('fishing_spots', []))
            )
        
        # Add configuration version from YAML
        if 'version' in metadata_template:
            metadata['config_version'] = str(metadata_template['version'])
        
        return metadata if metadata else None