#!/usr/bin/env python3
# Magic Animal: Coral
"""
WeeWX Surf & Fishing Forecast Extension Installer
Phase II: Local Surf & Fishing Forecast System

Copyright 2025 Shane Burkhardt
"""

import os
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
        print("WaveWatch III forecast data requires GRIB file processing capability.")
        print()
        
        if self.detect_grib_libraries():
            print(f"  {CORE_ICONS['status']} GRIB library detected: {self.available_library}")
            return True
        else:
            print(f"  {CORE_ICONS['warning']} No GRIB library found")
            print()
            print("PREREQUISITE MISSING:")
            print("This extension requires either eccodes-python or pygrib to process")
            print("WaveWatch III GRIB forecast data.")
            print()
            print("Installation options:")
            print("  1. Debian/Ubuntu: sudo apt-get install libeccodes0 libeccodes-dev python3-eccodes")
            print("  2. Alternative:   sudo apt-get install python3-grib")
            print("  3. pip fallback:  pip install eccodes-python")
            print("  4. pip fallback:  pip install pygrib")
            print()
            print("Please install a GRIB library and run the installer again.")
            print("See README.md for detailed installation instructions.")
            sys.exit(1)


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
        self.config_dict = config_dict
        self.yaml_data = yaml_data
        self.progress = InstallationProgressManager()
        self.grib_manager = GRIBLibraryManager()
        
    def run_interactive_setup(self):
        """Main configuration workflow"""
        
        print(f"{CORE_ICONS['navigation']} Surf & Fishing Forecast Configuration")
        print("="*60)
        print("Configure your personal surf and fishing forecast system")
        print("This extension reads data from Phase I and adds forecasting capabilities")
        print()
        
        # Step 1: Check dependencies
        self._check_phase_i_dependency()
        
        # Step 2: Install GRIB libraries
        grib_available = self._setup_grib_processing()
        
        # Step 3: Configure data sources
        data_sources = self._configure_data_sources()
        
        # Step 4: Configure locations
        locations = self._configure_locations()
        
        # Step 5: Transform to weewx.conf format
        config_dict = self._transform_to_weewx_conf(data_sources, locations, grib_available)
        
        return config_dict, locations
    
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
        """Configure atmospheric data strategy"""
        
        print(f"\n{CORE_ICONS['selection']} Atmospheric Data Strategy")
        print("Choose your atmospheric data sources for surf and fishing forecasts:")
        print()
        print("1. NOAA APIs Only (recommended for most users)")
        print("   - Uses NDBC buoys, CO-OPS tides, WaveWatch III")
        print("   - Most comprehensive offshore data")
        print()
        print("2. WeeWX Station + NOAA Supplement")
        print("   - Uses your weather station for local conditions")
        print("   - Supplements with NOAA marine data")
        print("   - Better local wind/pressure accuracy")
        print()
        
        while True:
            choice = input("Select data source (1-2): ").strip()
            if choice in ['1', '2']:
                break
            print(f"{CORE_ICONS['warning']} Please enter 1 or 2")
        
        if choice == '1':
            return {'type': 'noaa_only'}
        else:
            return self._configure_station_integration()
    
    def _configure_station_integration(self):
        """Configure which station sensors to use"""
        
        print(f"\n{CORE_ICONS['selection']} Station Sensor Integration")
        print("Select which station sensors to use (supplement with NOAA):")
        print()
        
        available_sensors = {
            'wind': 'Wind speed and direction',
            'pressure': 'Barometric pressure', 
            'temperature': 'Air temperature'
        }
        
        selected_sensors = {}
        for sensor_key, description in available_sensors.items():
            use_sensor = input(f"Use station {description}? (y/n, default n): ").strip().lower()
            selected_sensors[sensor_key] = use_sensor == 'y'
            
        return {
            'type': 'station_supplement',
            'sensors': selected_sensors
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
        """Configure surf-specific locations"""
        
        surf_spots = []
        print(f"\n{CORE_ICONS['selection']} Surf Spot Configuration")
        print("Enter your surf spots (max 5 spots)")
        print()
        
        while len(surf_spots) < 5:
            print(f"Surf Spot {len(surf_spots) + 1}:")
            
            name = input("  Spot name (e.g., 'Malibu', 'Ocean Beach') [Enter to finish]: ").strip()
            if not name:
                break
                
            # Get coordinates
            lat, lon = self._get_coordinates_with_validation(name)
            if not lat or not lon:
                continue
            
            # Get surf-specific characteristics
            spot_config = self._configure_surf_characteristics(name, lat, lon)
            surf_spots.append(spot_config)
            
            print(f"  {CORE_ICONS['status']} Added surf spot: {name}")
        
        return surf_spots
    
    def _configure_fishing_spots(self):
        """Configure fishing-specific locations"""
        
        fishing_spots = []
        print(f"\n{CORE_ICONS['selection']} Fishing Spot Configuration")
        print("Enter your fishing spots (max 5 spots)")
        print()
        
        while len(fishing_spots) < 5:
            print(f"Fishing Spot {len(fishing_spots) + 1}:")
            
            name = input("  Spot name (e.g., 'Santa Monica Pier', 'Newport Harbor') [Enter to finish]: ").strip()
            if not name:
                break
                
            # Get coordinates
            lat, lon = self._get_coordinates_with_validation(name)
            if not lat or not lon:
                continue
            
            # Get fishing-specific characteristics
            spot_config = self._configure_fishing_characteristics(name, lat, lon)
            fishing_spots.append(spot_config)
            
            print(f"  {CORE_ICONS['status']} Added fishing spot: {name}")
        
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
            selected_category = category_keys[0] if category_keys else 'mixed_bag'
        
        config['target_category'] = selected_category
        
        return config
    
    def _transform_to_weewx_conf(self, data_sources, locations, grib_available):
        """Transform configuration to weewx.conf format"""
        
        config_dict = {
            'SurfFishingService': {
                'enable': 'true',
                'forecast_interval': '21600',  # 6 hours
                
                # Data source configuration
                'station_integration': {
                    'type': data_sources['type']
                },
                
                # GRIB processing capability
                'grib_processing': {
                    'available': 'true',  # Always true if we reach this point
                    'library': self.grib_manager.available_library
                },

                # WaveWatch III endpoints from YAML
                'wavewatch_endpoints': self.yaml_data.get('api_endpoints', {}).get('wavewatch_iii', {}),
                
                # Fish categories from YAML
                'fish_categories': self.yaml_data.get('fish_categories', {}),
                
                # User locations
                'surf_spots': {},
                'fishing_spots': {}
            }
        }
        
        # Add station sensor configuration if using station integration
        if data_sources['type'] == 'station_supplement':
            config_dict['SurfFishingService']['station_integration']['sensors'] = data_sources.get('sensors', {})
        
        # Add user locations to config
        surf_count = 0
        fishing_count = 0
        
        for location in locations.get('surf_spots', []):
            spot_key = f"spot_{surf_count}"
            config_dict['SurfFishingService']['surf_spots'][spot_key] = {
                'name': location['name'],
                'latitude': str(location['latitude']),
                'longitude': str(location['longitude']),
                'bottom_type': location['bottom_type'],
                'exposure': location['exposure']
            }
            surf_count += 1
            
        for location in locations.get('fishing_spots', []):
            spot_key = f"spot_{fishing_count}"
            config_dict['SurfFishingService']['fishing_spots'][spot_key] = {
                'name': location['name'],
                'latitude': str(location['latitude']),
                'longitude': str(location['longitude']),
                'location_type': location['location_type'],
                'target_category': location['target_category']
            }
            fishing_count += 1
    
        return config_dict


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
        """Load YAML configuration data for installation"""
        
        yaml_file = 'bin/user/surf_fishing_fields.yaml'
        
        try:
            if os.path.exists(yaml_file):
                with open(yaml_file, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Default minimal configuration if YAML not found
                return {
                    'api_endpoints': {
                        'wavewatch_iii': {
                            'base_url': 'https://nomads.ncep.noaa.gov/pub/data/nccf/com/wave/prod/',
                            'grids': {
                                'us_east': {'bounds': [24, 50, -95, -65]},
                                'us_west': {'bounds': [24, 50, -130, -95]},
                                'glo_30m': {'bounds': [-90, 90, -180, 180]}
                            }
                        }
                    },
                    'fish_categories': {
                        'saltwater_inshore': {
                            'display_name': 'Saltwater Inshore',
                            'species': ['Striped Bass', 'Redfish', 'Snook'],
                            'pressure_preference': 'falling'
                        },
                        'mixed_bag': {
                            'display_name': 'Mixed Species',
                            'species': ['Various'],
                            'pressure_preference': 'stable'
                        }
                    }
                }
        except Exception as e:
            print(f"Warning: Could not load YAML data: {e}")
            return {}
    
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
        """Create database tables for surf and fishing forecasts"""
        
        print(f"\n{CORE_ICONS['selection']} Creating Database Tables")
        
        try:
            # Get database manager using WeeWX 5.1 patterns
            with weewx.manager.open_manager_with_config(config_dict, 'wx_binding') as manager:
                
                # Create surf spots table
                manager.connection.execute("""
                    CREATE TABLE IF NOT EXISTS marine_forecast_surf_spots (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        bottom_type TEXT,             -- sand, reef, point, jetty, mixed
                        exposure TEXT,                -- exposed, semi_protected, protected
                        preferred_swell_direction REAL, -- Optimal swell direction (degrees)
                        notes TEXT,                   -- User observations
                        created_date INTEGER,        -- Unix timestamp
                        active BOOLEAN DEFAULT 1     -- Enable/disable forecasting
                    )
                """)
                
                # Create fishing spots table
                manager.connection.execute("""
                    CREATE TABLE IF NOT EXISTS marine_forecast_fishing_spots (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL UNIQUE,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        location_type TEXT,           -- shore, pier, boat, mixed
                        target_category TEXT,         -- saltwater_inshore, freshwater_sport, etc.
                        structure_type TEXT,          -- sandy, rocky, pier, kelp, mixed
                        target_species TEXT,          -- JSON array of target species
                        notes TEXT,                   -- User observations
                        created_date INTEGER,        -- Unix timestamp
                        active BOOLEAN DEFAULT 1     -- Enable/disable forecasting
                    )
                """)
                
                # Create surf forecasts table
                manager.connection.execute("""
                    CREATE TABLE IF NOT EXISTS marine_forecast_surf_data (
                        id INTEGER PRIMARY KEY,
                        spot_id INTEGER NOT NULL,
                        forecast_time INTEGER NOT NULL,       -- Unix timestamp of forecast period
                        generated_time INTEGER NOT NULL,      -- When forecast was created
                        wave_height_min REAL,                -- Minimum expected wave height (feet)
                        wave_height_max REAL,                -- Maximum expected wave height (feet)
                        wave_period REAL,                    -- Dominant wave period (seconds)
                        wave_direction REAL,                 -- Wave direction (degrees)
                        wind_speed REAL,                     -- Wind speed (mph)
                        wind_direction REAL,                 -- Wind direction (degrees)
                        wind_condition TEXT,                 -- 'offshore', 'onshore', 'cross', 'calm'
                        tide_height REAL,                    -- Tide level (feet above datum)
                        tide_stage TEXT,                     -- 'rising', 'falling', 'high', 'low'
                        quality_rating INTEGER,              -- 1-5 star rating
                        confidence REAL,                     -- 0-1 confidence level
                        conditions_text TEXT,                -- "Good", "Fair", "Poor", "Excellent"
                        FOREIGN KEY(spot_id) REFERENCES marine_forecast_surf_spots(id),
                        UNIQUE(spot_id, forecast_time)
                    )
                """)
                
                # Create fishing forecasts table
                manager.connection.execute("""
                    CREATE TABLE IF NOT EXISTS marine_forecast_fishing_data (
                        id INTEGER PRIMARY KEY,
                        spot_id INTEGER NOT NULL,
                        forecast_date INTEGER NOT NULL,       -- Unix timestamp of forecast day (midnight)
                        period_name TEXT NOT NULL,            -- 'early_morning', 'morning', 'midday', 'afternoon', 'evening', 'night'
                        period_start_hour INTEGER,            -- 0-23 hour
                        period_end_hour INTEGER,              -- 0-23 hour
                        generated_time INTEGER NOT NULL,      -- When forecast was created
                        pressure_trend TEXT,                 -- 'falling', 'rising', 'stable'
                        pressure_change REAL,                -- 3-hour pressure change (inches Hg)
                        tide_movement TEXT,                  -- 'incoming', 'outgoing', 'slack'
                        species_activity TEXT,               -- 'high', 'moderate', 'low'
                        activity_rating INTEGER,             -- 1-5 star rating
                        conditions_text TEXT,                -- "Excellent", "Good", "Fair", "Poor"
                        best_species TEXT,                   -- JSON array of most active species
                        FOREIGN KEY(spot_id) REFERENCES marine_forecast_fishing_spots(id),
                        PRIMARY KEY (spot_id, forecast_date, period_name)
                    )
                """)
                
                # Create indexes for query performance
                manager.connection.execute("CREATE INDEX IF NOT EXISTS idx_surf_spots_active ON marine_forecast_surf_spots(active)")
                manager.connection.execute("CREATE INDEX IF NOT EXISTS idx_fishing_spots_active ON marine_forecast_fishing_spots(active)")
                manager.connection.execute("CREATE INDEX IF NOT EXISTS idx_surf_forecasts_time ON marine_forecast_surf_data(spot_id, forecast_time)")
                manager.connection.execute("CREATE INDEX IF NOT EXISTS idx_fishing_forecasts_date ON marine_forecast_fishing_data(spot_id, forecast_date)")
                
                # Insert user locations into database
                self._insert_user_locations(manager, selected_locations)
                
            print(f"  {CORE_ICONS['status']} Database tables created successfully")
            
        except Exception as e:
            print(f"  {CORE_ICONS['warning']} Error creating tables: {e}")
            raise
    
    def _insert_user_locations(self, manager, selected_locations):
        """Insert user-configured locations into database"""
        
        current_time = int(time.time())
        
        # Insert surf spots
        for spot in selected_locations.get('surf_spots', []):
            manager.connection.execute("""
                INSERT OR REPLACE INTO marine_forecast_surf_spots 
                (name, latitude, longitude, bottom_type, exposure, created_date, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                spot['name'],
                spot['latitude'],
                spot['longitude'],
                spot['bottom_type'],
                spot['exposure'],
                current_time,
                1
            ))
        
        # Insert fishing spots
        for spot in selected_locations.get('fishing_spots', []):
            manager.connection.execute("""
                INSERT OR REPLACE INTO marine_forecast_fishing_spots 
                (name, latitude, longitude, location_type, target_category, created_date, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                spot['name'],
                spot['latitude'],
                spot['longitude'],
                spot['location_type'],
                spot['target_category'],
                current_time,
                1
            ))
        
        # Commit the changes
        manager.connection.commit()
        
        surf_count = len(selected_locations.get('surf_spots', []))
        fishing_count = len(selected_locations.get('fishing_spots', []))
        
        print(f"  {CORE_ICONS['status']} Inserted {surf_count} surf spots and {fishing_count} fishing spots")